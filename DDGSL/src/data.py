"""Dataset-independent CSV loading, chronological segments and next-step windows."""
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.ddgsl_losses import WindowMetadata


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_checked(path, **kwargs):
    """Reject ambiguous raw headers before pandas can rename duplicate columns."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    names = [name.strip() for name in header]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError('CSV headers must be nonempty and unique after trimming')
    return pd.read_csv(path, **kwargs)


def load_series(path, *, feature_columns=None, timestamp_column=None,
                sequence_id_column=None, label_column=None, read_labels=False):
    """Return row-aligned features and contiguous segments; never infer a label."""
    frame = read_csv_checked(path)
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError('CSV must contain rows and unique column names')
    metadata = [c for c in (timestamp_column, sequence_id_column, label_column) if c]
    for name in metadata:
        if name not in frame:
            # A missing optional label is allowed for a normal training file.
            if name != label_column:
                raise ValueError('missing configured metadata column: ' + name)
    if feature_columns is None:
        selected = [c for c in frame if c not in metadata]
        if any(str(c).lower().startswith('unnamed:') for c in selected):
            raise ValueError('index-like column detected; set feature_columns explicitly')
    else:
        selected = list(feature_columns)
    if not selected or len(set(selected)) != len(selected) or any(c not in frame for c in selected):
        raise ValueError('invalid feature_columns')
    if set(selected) & set(metadata):
        raise ValueError('metadata columns cannot also be features')
    for name in selected:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError('feature must be numeric: ' + str(name))
    values = frame[selected].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('feature matrix has missing or nonfinite values')
    if sequence_id_column:
        ids = frame[sequence_id_column]
        if ids.isna().any():
            raise ValueError('missing sequence ID')
        ids = ids.astype(str).to_numpy()
    else:
        ids = np.repeat('single_sequence', len(frame))
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(frame)]
    segments = [{'start': int(start), 'stop': int(stop), 'sequence_id': str(ids[start]),
                 'segment_id': index}
                for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
    if timestamp_column:
        stamps = frame[timestamp_column]
        if stamps.isna().any():
            raise ValueError('missing timestamp')
        for segment in segments:
            local = stamps.iloc[segment['start']:segment['stop']]
            if not local.is_monotonic_increasing or local.duplicated().any():
                raise ValueError('timestamps must be strictly ordered within each segment')
    labels = None
    if read_labels and label_column and label_column in frame:
        labels = binary_labels(frame[label_column].to_numpy(), len(frame))
    return {'values': values, 'feature_columns': selected, 'segments': segments,
            'labels': labels, 'rows': len(frame), 'sha256': file_sha256(path)}


def binary_labels(values, rows):
    labels = np.asarray(values)
    if labels.shape != (rows,) or pd.isna(labels).any() or not np.isin(labels, [0, 1]).all():
        raise ValueError('labels must be row-aligned binary 0/1')
    return labels.astype(np.uint8)


def load_test_labels(series, test_path, label_path, label_column,
                     timestamp_column=None, sequence_id_column=None):
    """Called only in final evaluation, after the saved threshold is loaded."""
    if label_path:
        frame = read_csv_checked(label_path)
        if len(frame) != series['rows']:
            raise ValueError('test label file row count mismatch')
        options = [c for c in frame if c not in (timestamp_column, sequence_id_column)]
        name = label_column if label_column in frame else (options[0] if len(options) == 1 else None)
        if name is None:
            raise ValueError('test label column ambiguous')
        matching_columns = [c for c in (timestamp_column, sequence_id_column) if c]
        source = read_csv_checked(test_path, usecols=matching_columns) if matching_columns else None
        for column in (timestamp_column, sequence_id_column):
            if column and column in frame and not frame[column].equals(source[column]):
                raise ValueError('test labels and features have different row metadata')
        return binary_labels(frame[name].to_numpy(), series['rows'])
    if label_column:
        if label_column not in read_csv_checked(test_path, nrows=0).columns:
            return None
        frame = read_csv_checked(test_path, usecols=[label_column])
        return binary_labels(frame[label_column].to_numpy(), series['rows'])
    return None


def chronological_split(segments, ratio, window):
    """Split each known continuous sequence before making windows."""
    train, validation = [], []
    for segment in segments:
        start, stop = segment['start'], segment['stop']
        cut = start + int((stop - start) * ratio)
        if cut - start <= window or stop - cut <= window:
            raise ValueError('each train/validation segment must exceed window length')
        base = segment['segment_id'] * 2
        train.append(dict(segment, stop=cut, segment_id=base))
        validation.append(dict(segment, start=cut, segment_id=base + 1))
    return train, validation


def fit_scaler(values, train_segments, already_normalized):
    if already_normalized:
        return {'mode': 'identity_already_normalized', 'fit_segments': train_segments}
    train_values = np.concatenate([values[s['start']:s['stop']] for s in train_segments])
    return {'mode': 'train_minmax', 'minimum': train_values.min(axis=0).tolist(),
            'maximum': train_values.max(axis=0).tolist(), 'fit_segments': train_segments}


def transform(values, scaler):
    if scaler['mode'] == 'identity_already_normalized':
        output = np.asarray(values, dtype=np.float32)
    elif scaler['mode'] == 'train_minmax':
        low = np.asarray(scaler['minimum'], dtype=np.float64)
        high = np.asarray(scaler['maximum'], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(low) or low.shape != high.shape:
            raise ValueError('feature/scaler dimension mismatch')
        output = ((values-low)/np.where(high > low, high-low, 1.)).astype(np.float32)
    else:
        raise ValueError('unknown scaler mode')
    if not np.isfinite(output).all():
        raise ValueError('nonfinite scaled values')
    return output


class WindowDataset(Dataset):
    """One target t uses rows [t-w,t); its graph predecessor targets t-1."""
    def __init__(self, values, segments, *, window, stride, labels=None):
        if window < 1 or stride < 1 or values.ndim != 2:
            raise ValueError('invalid window settings')
        self.values = torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32)
        self.window = int(window)
        self.segments = segments
        self.specs = []
        for segment in segments:
            start, stop = segment['start'], segment['stop']
            if start < 0 or stop > len(values) or stop-start <= window:
                raise ValueError('invalid or too-short segment')
            self.specs.extend((int(t), segment) for t in range(start+window, stop, stride))
        if not self.specs:
            raise ValueError('no windows')
        self.targets = np.array([t for t, _ in self.specs], dtype=np.int64)
        self.labels = None if labels is None else binary_labels(labels, len(values))

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, index):
        t, segment = self.specs[index]
        first_t = segment['start'] + self.window
        first = t == first_t
        prior = t if first else t - 1
        def metadata(target, first_flag):
            return WindowMetadata(
                sequence_id=torch.tensor([segment['segment_id']], dtype=torch.long),
                segment_id=torch.tensor([segment['segment_id']], dtype=torch.long),
                window_index=torch.tensor([target-first_t], dtype=torch.long),
                time_index=torch.tensor([target], dtype=torch.long),
                stride=torch.tensor([1], dtype=torch.long),
                is_first=torch.tensor([first_flag], dtype=torch.bool))
        return {'x': self.values[t-self.window:t].T.contiguous(),
                'target': self.values[t],
                'previous_x': self.values[prior-self.window:prior].T.contiguous(),
                'metadata': metadata(t, first),
                'previous_metadata': metadata(prior, prior == first_t),
                'time_index': t,
                'label': -1 if self.labels is None else int(self.labels[t])}


def collate_windows(samples):
    def metadata(key):
        return WindowMetadata(**{field: torch.cat([getattr(item[key], field) for item in samples])
                                 for field in WindowMetadata.__dataclass_fields__})
    return {'x': torch.stack([item['x'] for item in samples]),
            'target': torch.stack([item['target'] for item in samples]),
            'previous_x': torch.stack([item['previous_x'] for item in samples]),
            'metadata': metadata('metadata'),
            'previous_metadata': metadata('previous_metadata'),
            'time_index': torch.tensor([item['time_index'] for item in samples]),
            'label': torch.tensor([item['label'] for item in samples])}


def target_manifest(dataset, stride):
    ranges = [{'start': s['start'] + dataset.window, 'stop': s['stop'], 'step': stride,
               'segment_id': s['segment_id']} for s in dataset.segments]
    digest = hashlib.sha256(dataset.targets.astype('<i8').tobytes()).hexdigest()
    return {'ranges': ranges, 'count': len(dataset), 'sha256_le_int64': digest}
