"""Paper Section 4.3.2: validation-only residual calibration and point metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score


def calibrate(validation_prediction, validation_target, *, eps=1e-8, rho=0.995):
    prediction = np.asarray(validation_prediction, dtype=np.float64)
    target = np.asarray(validation_target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[0] == 0:
        raise ValueError('validation predictions and targets must be matching [time,node] arrays')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('non-finite validation inputs')
    if eps <= 0 or not 0 < rho < 1:
        raise ValueError('eps must be positive and rho within (0,1)')
    residual = np.abs(prediction - target)
    med = np.median(residual, axis=0)
    mad = np.median(np.abs(residual - med), axis=0)
    score = np.max((residual - med) / (mad + eps), axis=1)
    theta = float(np.quantile(score, rho))
    return {'median': med, 'mad': mad, 'eps': eps, 'rho': rho,
            'theta': theta, 'validation_scores': score}


def score_test(prediction, target, calibration):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError('test prediction and target shapes differ')
    if prediction.shape[1] != len(calibration['median']):
        raise ValueError('test node count differs from validation')
    residual = np.abs(prediction - target)
    scores = np.max((residual - calibration['median']) /
                    (calibration['mad'] + calibration['eps']), axis=1)
    if not np.isfinite(scores).all():
        raise FloatingPointError('non-finite test anomaly score')
    return scores, (scores > calibration['theta']).astype(np.uint8)


def point_metrics(labels, scores, predictions):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    predictions = np.asarray(predictions)
    if labels.ndim != 1 or labels.shape != scores.shape or labels.shape != predictions.shape:
        raise ValueError('point arrays must align exactly')
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError('labels and predictions must be binary')
    if not np.isfinite(scores).all():
        raise ValueError('non-finite scores')
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auroc = float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'precision': precision, 'recall': recall, 'f1': f1, 'auroc': auroc,
            'evaluated_points': int(len(labels))}
