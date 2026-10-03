"""Explicit CSV preparation; normalization and windows remain in train.py.

Block medians and binary block-maximum labels follow the SWaT script.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.data import binary_labels, file_sha256


def read_input(path, features, label_column, timestamp_column, sequence_id_column,
               missing):
    """Validate explicit features and preserve contiguous input segments."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    names = [name.strip() for name in header]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError('CSV headers must be nonempty and unique after trimming')
    frame = pd.read_csv(path, encoding='utf-8-sig')
    frame.columns = names
    if frame.empty:
        raise ValueError('input CSV is empty')
    metadata = [c for c in (label_column, timestamp_column, sequence_id_column) if c]
    if len(set(metadata)) != len(metadata) or set(features) & set(metadata):
        raise ValueError('feature and metadata columns must be distinct')
    required = features + [c for c in (timestamp_column, sequence_id_column) if c]
    if any(c not in frame for c in required):
        raise ValueError('missing requested feature or metadata column')
    values = frame[features].apply(pd.to_numeric, errors='raise').astype(np.float64)
    if np.isinf(values.to_numpy()).any():
        raise ValueError('infinite feature values are not supported')
    if sequence_id_column:
        if frame[sequence_id_column].isna().any():
            raise ValueError('missing sequence ID')
        ids = frame[sequence_id_column].astype(str).to_numpy()
    else:
        ids = np.repeat('single_sequence_assumed', len(frame))
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(frame)]
    segments = []
    missing_counts = values.isna().sum().astype(int).to_dict()
    for number, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        start, stop = int(start), int(stop)
        if timestamp_column:
            stamps = frame[timestamp_column].iloc[start:stop]
            if stamps.isna().any() or stamps.duplicated().any() or not stamps.is_monotonic_increasing:
                raise ValueError('timestamps must be strictly ordered within each segment')
        if missing == 'forward-fill':
            values.iloc[start:stop] = values.iloc[start:stop].ffill()
        segments.append({'segment_id': number, 'source_sequence_id': str(ids[start]),
                         'start': start, 'stop': stop})
    if values.isna().any().any():
        raise ValueError('missing features remain; forward-fill cannot fill segment-leading gaps')
    labels = None
    if label_column in frame:
        labels = binary_labels(frame[label_column].to_numpy(), len(frame))
    return frame, values.to_numpy(), labels, segments, missing_counts


def reduce_series(frame, values, labels, segments, features, timestamp_column,
                  downsample, tail, prefix_blocks):
    """Aggregate within segments; record every retained source row interval."""
    if prefix_blocks and len(segments) != 1:
        raise ValueError('prefix trimming requires a single explicitly understood train segment')
    rows, mappings, summaries = [], [], []
    for segment in segments:
        start, stop = segment['start'], segment['stop']
        remainder = (stop-start) % downsample
        if remainder and tail == 'error':
            raise ValueError('incomplete final block; explicitly choose --tail keep or drop')
        block_stop = stop-remainder if tail == 'drop' else stop
        starts = list(range(start, block_stop, downsample))
        selected = starts[prefix_blocks:]
        if not selected:
            raise ValueError('processing would remove an entire segment')
        output_start = len(rows)
        for left in selected:
            right = min(left+downsample, stop)
            with np.errstate(over='ignore', invalid='ignore'):
                aggregated = np.median(values[left:right], axis=0)
            if not np.isfinite(aggregated).all():
                raise ValueError('nonfinite block median; check input numeric magnitudes')
            row = dict(zip(features, aggregated.tolist()))
            # Unique contiguous-segment IDs prevent separated source IDs from merging.
            row['segment_id'] = segment['segment_id']
            if timestamp_column:
                row['timestamp'] = frame[timestamp_column].iloc[right-1]
            if labels is not None:
                row['label'] = int(labels[left:right].max())
            mappings.append({'output_row': len(rows), 'segment_id': segment['segment_id'],
                             'source_row_start': left, 'source_row_stop': right})
            rows.append(row)
        summaries.append(dict(segment, output_start=output_start, output_stop=len(rows),
            dropped_prefix_rows=selected[0]-start, dropped_tail_rows=stop-block_stop))
    return pd.DataFrame(rows), pd.DataFrame(mappings), summaries


def prepare_data(train_path, test_path, output_dir, *, feature_columns,
                 label_column='label', timestamp_column=None, sequence_id_column=None,
                 downsample=1, missing='error', tail='error', drop_train_prefix_blocks=0,
                 already_normalized=False):
    """Write a new directory only; never edit source CSVs or invent labels."""
    features = list(feature_columns)
    if not features or len(set(features)) != len(features) or any(not isinstance(c, str) or not c for c in features):
        raise ValueError('feature_columns must be explicit, nonempty and unique')
    if set(features) & {'label', 'segment_id', 'timestamp'}:
        raise ValueError('features cannot use reserved output metadata names')
    if type(downsample) is not int or downsample < 1 or type(drop_train_prefix_blocks) is not int or drop_train_prefix_blocks < 0:
        raise ValueError('downsample must be positive and prefix blocks nonnegative integers')
    if missing not in ('error', 'forward-fill') or tail not in ('error', 'keep', 'drop'):
        raise ValueError('unknown missing-value or tail rule')
    if type(already_normalized) is not bool or not isinstance(label_column, str) or not label_column:
        raise ValueError('invalid normalization flag or label column')
    if Path(train_path).resolve() == Path(test_path).resolve():
        raise ValueError('train and test must be separate source files')
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('output directory already exists; choose a new directory')
    options = dict(feature_columns=features, label_column=label_column,
        timestamp_column=timestamp_column, sequence_id_column=sequence_id_column,
        downsample=downsample, missing=missing, tail=tail,
        drop_train_prefix_blocks=drop_train_prefix_blocks,
        already_normalized=already_normalized)
    sources, prepared = {}, {}
    for name, path in (('train', train_path), ('test', test_path)):
        path = Path(path).resolve()
        before_hash = file_sha256(path)
        frame, values, labels, segments, missing_counts = read_input(
            path, features, label_column, timestamp_column, sequence_id_column, missing)
        if name == 'train' and labels is not None and labels.any():
            raise ValueError('training input must be normal; anomalous rows are not silently removed')
        if name == 'test' and labels is None:
            raise ValueError('test input requires the specified binary label column')
        reduced, mapping, summaries = reduce_series(frame, values, labels, segments,
            features, timestamp_column, downsample, tail,
            drop_train_prefix_blocks if name == 'train' else 0)
        if file_sha256(path) != before_hash:
            raise ValueError('source file changed while preprocessing')
        sources[name] = {'path': str(path), 'sha256': before_hash, 'rows': len(frame),
            'input_columns': list(frame.columns),
            'unused_columns': [c for c in frame if c not in features and c not in
                               (label_column, timestamp_column, sequence_id_column)],
            'missing_before': missing_counts, 'segments': summaries,
            'labels_present': labels is not None,
            'output_rows': len(reduced),
            'feature_min_after_fill': dict(zip(features, values.min(axis=0).tolist())),
            'feature_max_after_fill': dict(zip(features, values.max(axis=0).tolist()))}
        prepared[name] = reduced, mapping
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, (frame, mapping) in prepared.items():
        frame.to_csv(output_dir/(name+'.csv'), index=False)
        mapping.to_csv(output_dir/(name+'_row_mapping.csv'), index=False)
    (output_dir/'list.txt').write_text('\n'.join(features)+'\n', encoding='utf-8')
    manifest = {'schema_version': 1, 'procedure': 'new_explicit_csv_preparation',
        'historical_paper_provenance_verified': False,
        'options': options, 'source_files': sources,
        'output_sha256': {p.name: file_sha256(p) for p in sorted(output_dir.iterdir())},
        'implementation_sha256': file_sha256(__file__),
        'data_helpers_sha256': file_sha256(Path(__file__).parent/'src'/'data.py'),
        'environment': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__},
        'index_convention': 'zero-based data rows excluding CSV header; stop is exclusive',
        'aggregation': 'nonoverlapping within-segment median; binary label maximum',
        'timestamp_rule': 'last source timestamp in block; no resampling or gap inference',
        'normalization': 'none here; train.py fits training-subset minmax unless normalized=true',
        'boundary_evidence': 'provided contiguous IDs' if sequence_id_column else 'single segment assumed per input',
        'dataset_config': {'train_path': str((output_dir/'train.csv').resolve()),
            'test_path': str((output_dir/'test.csv').resolve()), 'test_label_path': None,
            'feature_columns': features, 'timestamp_column': 'timestamp' if timestamp_column else None,
            'sequence_id_column': 'segment_id', 'label_column': 'label',
            'normalized': already_normalized, 'train_ratio': 0.7}}
    (output_dir/'preprocessing_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output-dir', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--features', nargs='+', help='explicit ordered numeric feature names')
    group.add_argument('--feature-list', help='UTF-8 file, one feature per line in model order')
    parser.add_argument('--label-column', default='label', help='existing binary 0/1 column')
    parser.add_argument('--timestamp-column')
    parser.add_argument('--sequence-id-column')
    parser.add_argument('--downsample', type=int, default=1)
    parser.add_argument('--missing', choices=['error', 'forward-fill'], default='error')
    parser.add_argument('--tail', choices=['error', 'keep', 'drop'], default='error')
    parser.add_argument('--drop-train-prefix-blocks', type=int, default=0)
    parser.add_argument('--already-normalized', action='store_true',
                        help='declare upstream scaling; this program does not normalize')
    args = parser.parse_args()
    features = args.features
    if args.feature_list:
        features = [line.strip() for line in Path(args.feature_list).read_text(
            encoding='utf-8-sig').splitlines() if line.strip()]
    manifest = prepare_data(args.train, args.test, args.output_dir,
        feature_columns=features, label_column=args.label_column,
        timestamp_column=args.timestamp_column, sequence_id_column=args.sequence_id_column,
        downsample=args.downsample, missing=args.missing, tail=args.tail,
        drop_train_prefix_blocks=args.drop_train_prefix_blocks,
        already_normalized=args.already_normalized)
    print('Prepared {} train and {} test rows; {} features. No training performed.'.format(
        manifest['source_files']['train']['output_rows'],
        manifest['source_files']['test']['output_rows'], len(features)))
    print('Copy dataset_config from preprocessing_manifest.json into configs/default.yaml; paths are absolute.')


if __name__ == '__main__':
    main()
