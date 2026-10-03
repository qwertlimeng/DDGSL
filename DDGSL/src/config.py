"""One-run YAML configuration, deterministic seed and artifact helpers."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from models.ddgsl_config import CoreConfig


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                  separators=(',', ':')).encode('utf-8')).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                          encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + ' must be a positive integer')


def _finite_number(value, name, *, positive=False, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(name + ' out of range')


def load_config(path, overrides=()):
    with Path(path).open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError('YAML root must be a mapping')
    config = copy.deepcopy(config)
    for assignment in overrides:
        if '=' not in assignment or '.' not in assignment.split('=', 1)[0]:
            raise ValueError('--set expects section.key=value')
        dotted, raw = assignment.split('=', 1)
        keys = dotted.split('.')
        location = config
        for key in keys[:-1]:
            if not isinstance(location, dict) or key not in location:
                raise ValueError('unknown config key: ' + dotted)
            location = location[key]
        if not isinstance(location, dict) or keys[-1] not in location:
            raise ValueError('unknown config key: ' + dotted)
        location[keys[-1]] = yaml.safe_load(raw)
    if set(config) != {'dataset', 'model', 'training', 'detection', 'output_dir'}:
        raise ValueError('configuration must have dataset/model/training/detection/output_dir')
    dataset, model, training, detection = (config[k] for k in
                                           ('dataset', 'model', 'training', 'detection'))
    if set(dataset) != {'train_path', 'test_path', 'test_label_path', 'feature_columns',
                        'timestamp_column', 'sequence_id_column', 'label_column',
                        'normalized', 'train_ratio'}:
        raise ValueError('invalid dataset configuration keys')
    for key in ('train_path', 'test_path'):
        if not isinstance(dataset[key], str) or not dataset[key]:
            raise ValueError(key + ' must be a nonempty path')
    for key in ('test_label_path', 'timestamp_column', 'sequence_id_column', 'label_column'):
        if dataset[key] is not None and not isinstance(dataset[key], str):
            raise ValueError(key + ' must be a string or null')
    if dataset['feature_columns'] is not None and (not isinstance(dataset['feature_columns'], list)
            or not dataset['feature_columns'] or
            not all(isinstance(c, str) for c in dataset['feature_columns'])):
        raise ValueError('feature_columns must be null or a nonempty string list')
    if type(dataset['normalized']) is not bool:
        raise ValueError('normalized must be boolean')
    _finite_number(dataset['train_ratio'], 'train_ratio', positive=True)
    if dataset['train_ratio'] >= 1:
        raise ValueError('train_ratio must be less than one')
    if set(model) != {'window_size', 'top_k', 'embedding_dim', 'stib_depth'}:
        raise ValueError('invalid model configuration keys')
    for key in model:
        _positive_int(model[key], key)
    if model['window_size'] < 2 ** model['stib_depth']:
        raise ValueError('window too short for recursive STIB depth')
    if set(training) != {'batch_size', 'train_stride', 'learning_rate', 'weight_decay',
                         'max_epochs', 'patience', 'mu1', 'mu2', 'seed', 'optimizer'}:
        raise ValueError('invalid training configuration keys')
    for key in ('batch_size', 'train_stride', 'max_epochs', 'patience'):
        _positive_int(training[key], key)
    for key in ('learning_rate', 'weight_decay', 'mu1', 'mu2'):
        _finite_number(training[key], key, positive=key == 'learning_rate',
                       nonnegative=key != 'learning_rate')
    if training['optimizer'] != 'Adam' or type(training['seed']) is not int or \
            not 0 <= training['seed'] <= 2**32-1:
        raise ValueError('optimizer must be Adam and seed must be uint32')
    if set(detection) != {'rho', 'epsilon', 'evaluation_stride'}:
        raise ValueError('invalid detection configuration keys')
    _finite_number(detection['rho'], 'rho', positive=True)
    if detection['rho'] >= 1:
        raise ValueError('rho must be below one')
    _finite_number(detection['epsilon'], 'epsilon', positive=True)
    _positive_int(detection['evaluation_stride'], 'evaluation_stride')
    if not isinstance(config['output_dir'], str) or not config['output_dir']:
        raise ValueError('output_dir must be a nonempty path')
    return config


def data_path(config_file, relative):
    if relative is None:
        return None
    path = Path(relative)
    return path if path.is_absolute() else (Path(config_file).resolve().parent / path).resolve()


def core_config(config, node_count):
    if node_count < 2 or config['model']['top_k'] >= node_count:
        raise ValueError('top_k must be less than the number of variables')
    model, train = config['model'], config['training']
    return CoreConfig.confirmed(model['window_size'], node_count,
        embedding_dim=model['embedding_dim'], topk=model['top_k'],
        stib_depth=model['stib_depth'], mu1=train['mu1'], mu2=train['mu2'])


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'):
        raise ValueError('invalid CUBLAS_WORKSPACE_CONFIG for deterministic CUDA')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return {'seed': seed, 'cudnn_deterministic': True, 'cudnn_benchmark': False,
            'deterministic_algorithms': True,
            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']}


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('DDGSL experiment commands require a CUDA GPU')
    return torch.device('cuda:0')


def source_hashes(root):
    root = Path(root)
    paths = [root/'train.py', root/'evaluate.py'] + sorted((root/'models').glob('*.py')) + \
        sorted((root/'src').glob('*.py'))
    return {str(path.relative_to(root)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}
