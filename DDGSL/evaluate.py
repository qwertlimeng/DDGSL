import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import uuid

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from models.ddgsl_config import CoreConfig
from models.ddgsl_model import DDGSL
from src.config import (data_path, digest, load_config, read_json, require_cuda,
                        seed_everything, source_hashes, write_json)
from src.data import (WindowDataset, file_sha256, load_series, load_test_labels,
                      target_manifest, transform)
from src.scoring import point_metrics, score_test
from train import ROOT, loader, predict


def evaluate_once(checkpoint_path, config_file, overrides=(), device=None):
    """The CLI requires CUDA; explicit CPU is only for tiny synthetic tests."""
    config = load_config(config_file, overrides)
    run_dir = Path(checkpoint_path).resolve().parent
    effective = read_json(run_dir/'effective_config.json')
    manifest = read_json(run_dir/'split_manifest.json')
    scaler = read_json(run_dir/'scaler.json')
    calibration = read_json(run_dir/'calibration.json')
    saved_source = read_json(run_dir/'source_sha256.json')
    if effective['status'] != 'complete' or config != effective['requested_config'] or \
            digest(config) != effective['config_id']:
        raise ValueError('evaluation configuration differs from completed training run')
    if digest({k: v for k, v in manifest.items() if k != 'manifest_id'}) != \
            manifest['manifest_id'] or manifest['manifest_id'] != effective['manifest_id']:
        raise ValueError('split manifest identity mismatch')
    if saved_source != source_hashes(ROOT):
        raise ValueError('source files changed since training')
    if digest(scaler) != effective['scaler_id'] or \
            digest(calibration) != effective['calibration_id']:
        raise ValueError('saved scaler or calibration changed since training')
    dataset = config['dataset']
    paths = {key: data_path(config_file, dataset[key+'_path']) for key in
             ('train', 'test', 'test_label')}
    current_hashes = {key: None if path is None else file_sha256(path)
                      for key, path in paths.items()}
    if current_hashes != manifest['file_sha256'] or \
            current_hashes != effective['file_sha256']:
        raise ValueError('data files changed since training')
    if device is None:
        device = require_cuda()
    seed_everything(config['training']['seed'])
    with Path(checkpoint_path).open('rb') as handle:
        checkpoint = torch.load(handle, map_location=device)
    if checkpoint['checkpoint_identity'] != effective['checkpoint_identity'] or \
            checkpoint['core_config'] != effective['core_config'] or \
            checkpoint['selected_epoch'] != effective['selected_epoch'] or \
            checkpoint['validation_mse'] != effective['validation_mse']:
        raise ValueError('checkpoint metadata does not match saved effective configuration')
    with (run_dir/'training_history.csv').open(encoding='utf-8', newline='') as stream:
        history = list(csv.DictReader(stream))
    if not history:
        raise ValueError('missing training history')
    selected = min(history, key=lambda row: float(row['validation_mse']))
    if int(selected['epoch']) != checkpoint['selected_epoch'] or \
            float(selected['validation_mse']) != checkpoint['validation_mse']:
        raise ValueError('checkpoint is not the minimum-validation-MSE epoch')
    model = DDGSL(CoreConfig(**checkpoint['core_config'])).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.eval()
    options = dict(feature_columns=manifest['feature_columns'],
                   timestamp_column=dataset['timestamp_column'],
                   sequence_id_column=dataset['sequence_id_column'],
                   label_column=dataset['label_column'])
    test = load_series(paths['test'], read_labels=False, **options)
    if test['rows'] != manifest['row_counts']['test'] or \
            test['segments'] != manifest['test_segments']:
        raise ValueError('test rows or continuous segments differ from training manifest')
    if scaler['mode'] == 'identity_already_normalized' and not dataset['normalized'] or \
            scaler['mode'] == 'train_minmax' and dataset['normalized']:
        raise ValueError('saved scaler mode differs from configuration')
    test_values = transform(test['values'], scaler)
    windows = WindowDataset(test_values, test['segments'],
        window=model.config.window_size, stride=config['detection']['evaluation_stride'])
    if target_manifest(windows, config['detection']['evaluation_stride']) != \
            manifest['target_indices']['test']:
        raise ValueError('test target indices differ from frozen split')
    batches = loader(windows, config['training']['batch_size'], False,
                     config['training']['seed'])
    prediction, target = predict(model, batches, device, windows.targets)
    frozen = {'median': np.asarray(calibration['median'], dtype=np.float64),
              'mad': np.asarray(calibration['mad'], dtype=np.float64),
              'eps': calibration['epsilon'], 'theta': calibration['theta']}
    if len(frozen['median']) != model.config.node_num or \
            calibration['rho'] != config['detection']['rho'] or \
            calibration['epsilon'] != config['detection']['epsilon']:
        raise ValueError('calibration differs from model or configuration')
    scores, predicted = score_test(prediction, target, frozen)
    # Only now, after the saved calibration and threshold are loaded, read labels.
    labels = load_test_labels(test, paths['test'], paths['test_label'],
                              dataset['label_column'], dataset['timestamp_column'],
                              dataset['sequence_id_column'])
    point_labels = None if labels is None else labels[windows.targets]
    metrics = None if point_labels is None else point_metrics(point_labels, scores, predicted)
    eval_id = 'evaluation_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + \
              '_' + uuid.uuid4().hex[:8]
    output = run_dir/eval_id
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'point_predictions.npz', time_index=windows.targets,
                        prediction=prediction, target=target, anomaly_score=scores,
                        predicted_label=predicted,
                        true_label=np.full(len(scores), -1, dtype=np.int8) if point_labels is None
                        else point_labels)
    write_json(output/'metrics.json', {'checkpoint_identity': effective['checkpoint_identity'],
        'threshold': calibration['theta'], 'evaluated_points': len(scores),
        'labels_available': point_labels is not None, 'metrics': metrics})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs'/'default.yaml')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='SECTION.KEY=VALUE')
    args = parser.parse_args()
    print(evaluate_once(args.checkpoint, args.config, args.overrides,
                        device=require_cuda()))


if __name__ == '__main__':
    main()
