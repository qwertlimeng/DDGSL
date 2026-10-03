import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import time
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.ddgsl_losses import core_objective
from models.ddgsl_model import DDGSL
from src.config import (core_config, data_path, digest, load_config, require_cuda,
                        seed_everything, source_hashes, write_json)
from src.data import (WindowDataset, chronological_split, collate_windows, file_sha256,
                      fit_scaler, load_series, target_manifest, transform)
from src.scoring import calibrate

ROOT = Path(__file__).resolve().parent


def loader(dataset, batch_size, shuffle, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=collate_windows,
                      generator=torch.Generator().manual_seed(seed))


def predict(model, batches, device, expected_times):
    model.eval()
    predictions, targets, times = [], [], []
    with torch.no_grad():
        for batch in batches:
            predictions.append(model(batch['x'].to(device)).cpu().numpy())
            targets.append(batch['target'].numpy())
            times.append(batch['time_index'].numpy())
    actual = np.concatenate(times)
    np.testing.assert_array_equal(actual, expected_times)
    return np.concatenate(predictions), np.concatenate(targets)


def loss_step(model, batch, device):
    details = model(batch['x'].to(device), return_details=True,
                    metadata=batch['metadata'].to(device),
                    previous_x=batch['previous_x'].to(device),
                    previous_metadata=batch['previous_metadata'].to(device))
    cfg = model.config
    terms = core_objective(details['prediction'], batch['target'].to(device),
                           details['L_smooth'], details['L_direction'], cfg.mu1, cfg.mu2)
    if not all(torch.isfinite(value).all().item() for value in terms.values()):
        raise FloatingPointError('nonfinite training loss')
    return terms


def prepare(config, config_file):
    dataset, model, training, detection = (config[key] for key in
                                           ('dataset', 'model', 'training', 'detection'))
    train_path = data_path(config_file, dataset['train_path'])
    test_path = data_path(config_file, dataset['test_path'])
    label_path = data_path(config_file, dataset['test_label_path'])
    options = dict(feature_columns=dataset['feature_columns'],
                   timestamp_column=dataset['timestamp_column'],
                   sequence_id_column=dataset['sequence_id_column'],
                   label_column=dataset['label_column'])
    normal = load_series(train_path, read_labels=True, **options)
    if normal['labels'] is not None and normal['labels'].any():
        raise ValueError('training CSV must contain only normal points')
    test = load_series(test_path, read_labels=False, **options)
    if normal['feature_columns'] != test['feature_columns']:
        raise ValueError('train/test feature column order differs')
    if label_path and not label_path.is_file():
        raise FileNotFoundError('configured test label file is missing')
    core = core_config(config, len(normal['feature_columns']))
    train_segments, val_segments = chronological_split(normal['segments'],
        dataset['train_ratio'], core.window_size)
    scaler = fit_scaler(normal['values'], train_segments, dataset['normalized'])
    scaled_train = transform(normal['values'], scaler)
    train_windows = WindowDataset(scaled_train, train_segments,
        window=core.window_size, stride=training['train_stride'])
    val_windows = WindowDataset(scaled_train, val_segments,
        window=core.window_size, stride=detection['evaluation_stride'])
    # Test is read without labels here, only to freeze row/segment/target metadata.
    test_windows = WindowDataset(np.zeros_like(test['values'], dtype=np.float32),
        test['segments'], window=core.window_size, stride=detection['evaluation_stride'])
    hashes = {'train': normal['sha256'], 'test': test['sha256'],
              'test_label': None if label_path is None else file_sha256(label_path)}
    manifest = {
        'file_sha256': hashes,
        'row_counts': {'train': normal['rows'], 'test': test['rows']},
        'feature_columns': normal['feature_columns'], 'node_count': core.node_num,
        'train_ratio': dataset['train_ratio'], 'window_size': core.window_size,
        'train_stride': training['train_stride'],
        'evaluation_stride': detection['evaluation_stride'],
        'smoothness_target_delta': 1,
        'segment_rule': 'contiguous sequence-ID runs; one segment if ID column absent',
        'train_segments': train_segments, 'validation_segments': val_segments,
        'test_segments': test['segments'],
        'target_indices': {
            'train': target_manifest(train_windows, training['train_stride']),
            'validation': target_manifest(val_windows, detection['evaluation_stride']),
            'test': target_manifest(test_windows, detection['evaluation_stride'])},
    }
    manifest['manifest_id'] = digest(manifest)
    return core, normal, test, scaler, train_windows, val_windows, manifest


def train_once(config_file, overrides=(), device=None):
    """The CLI requires CUDA; an explicit CPU device is for tiny synthetic tests only."""
    config = load_config(config_file, overrides)
    if device is None:
        device = require_cuda()
    seed_state = seed_everything(config['training']['seed'])
    core, normal, test, scaler, training_data, validation_data, manifest = \
        prepare(config, config_file)
    source = source_hashes(ROOT)
    config_id = digest(config)
    identity = digest({'config_id': config_id, 'manifest_id': manifest['manifest_id'],
                       'source_sha256': source})
    now = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = now + '_' + identity[:10] + '_' + uuid.uuid4().hex[:8]
    run_root = data_path(config_file, config['output_dir'])
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    train_cfg, detection = config['training'], config['detection']
    train_loader = loader(training_data, train_cfg['batch_size'], True, train_cfg['seed'])
    val_loader = loader(validation_data, train_cfg['batch_size'], False, train_cfg['seed'])
    model = DDGSL(core).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'],
                                 weight_decay=train_cfg['weight_decay'])
    effective = {'run_id': run_id, 'status': 'running', 'requested_config': config,
                 'config_id': config_id, 'core_config': core.to_dict(),
                 'manifest_id': manifest['manifest_id'], 'checkpoint_identity': identity,
                 'scaler_id': digest(scaler),
                 'file_sha256': manifest['file_sha256'],
                 'feature_columns': normal['feature_columns'],
                 'seed_state': seed_state, 'device': str(device),
                 'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                 'torch_version': torch.__version__, 'numpy_version': np.__version__,
                 'parameter_count': sum(p.numel() for p in model.parameters()),
                 'optimizer': 'Adam', 'learning_rate': optimizer.param_groups[0]['lr'],
                 'weight_decay': optimizer.param_groups[0]['weight_decay'],
                 'train_windows': len(training_data), 'validation_windows': len(validation_data)}
    write_json(run_dir/'effective_config.json', effective)
    write_json(run_dir/'split_manifest.json', manifest)
    write_json(run_dir/'scaler.json', scaler)
    write_json(run_dir/'source_sha256.json', source)
    started = time.perf_counter()
    best_mse, best_epoch, stale = float('inf'), None, 0
    history_file = run_dir/'training_history.csv'
    try:
        with history_file.open('w', encoding='utf-8', newline='') as stream:
            fields = ['epoch', 'training_mse', 'L_smooth', 'L_direction',
                      'total_loss', 'validation_mse', 'selected', 'epoch_seconds']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, train_cfg['max_epochs'] + 1):
                epoch_started = time.perf_counter()
                model.train()
                sums = {name: 0. for name in ('mse', 'smooth', 'direction', 'total')}
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    terms = loss_step(model, batch, device)
                    terms['total'].backward()
                    optimizer.step()
                    for name in sums:
                        sums[name] += float(terms[name].detach()) * len(batch['x'])
                val_prediction, val_target = predict(model, val_loader, device,
                                                       validation_data.targets)
                val_mse = float(np.mean((val_prediction-val_target)**2))
                if not np.isfinite(val_mse):
                    raise FloatingPointError('nonfinite validation MSE')
                selected = val_mse < best_mse
                if selected:
                    best_mse, best_epoch, stale = val_mse, epoch, 0
                    with (run_dir/'selected_model.pt').open('wb') as handle:
                        torch.save({'model_state': model.state_dict(),
                                    'core_config': core.to_dict(), 'selected_epoch': epoch,
                                    'validation_mse': val_mse,
                                    'checkpoint_identity': identity}, handle)
                else:
                    stale += 1
                writer.writerow({'epoch': epoch, 'training_mse': sums['mse']/len(training_data),
                    'L_smooth': sums['smooth']/len(training_data),
                    'L_direction': sums['direction']/len(training_data),
                    'total_loss': sums['total']/len(training_data),
                    'validation_mse': val_mse, 'selected': int(selected),
                    'epoch_seconds': time.perf_counter()-epoch_started})
                stream.flush()
                if stale >= train_cfg['patience']:
                    break
        with (run_dir/'selected_model.pt').open('rb') as handle:
            checkpoint = torch.load(handle, map_location=device)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        val_prediction, val_target = predict(model, val_loader, device,
                                               validation_data.targets)
        calibration = calibrate(val_prediction, val_target,
            eps=detection['epsilon'], rho=detection['rho'])
        saved_calibration = {
            'median': calibration['median'].tolist(), 'mad': calibration['mad'].tolist(),
            'epsilon': detection['epsilon'], 'rho': detection['rho'],
            'theta': calibration['theta'], 'validation_mse': best_mse}
        write_json(run_dir/'calibration.json', saved_calibration)
        np.savez_compressed(run_dir/'normal_validation_scores.npz',
                            time_index=validation_data.targets,
                            anomaly_score=calibration['validation_scores'])
        effective.update(status='complete', selected_epoch=best_epoch,
                         validation_mse=best_mse,
                         calibration_id=digest(saved_calibration),
                         training_seconds=time.perf_counter()-started)
        write_json(run_dir/'effective_config.json', effective)
    except Exception as error:
        effective.update(status='failed', error=repr(error),
                         training_seconds=time.perf_counter()-started)
        write_json(run_dir/'effective_config.json', effective)
        raise
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs'/'default.yaml')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='SECTION.KEY=VALUE')
    args = parser.parse_args()
    print(train_once(args.config, args.overrides, device=require_cuda()))


if __name__ == '__main__':
    main()
