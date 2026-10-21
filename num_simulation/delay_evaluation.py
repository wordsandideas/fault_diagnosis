"""Normal-only calibration and label-independent delay evaluation.

Indices in residual arrays refer to source time t; their target is a[t+1].
Use one evaluation_start and the same calibration/test data for every delay.
"""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score


def _predict(experiment, a, u, g, fault, pdf):
    length = len(a)
    if any(len(x) != length for x in (u, g, fault, pdf)):
        raise ValueError('All input sequences must have equal lengths')
    if any(not torch.isfinite(x).all() for x in (a, u, g, pdf)):
        raise ValueError('Observed inputs must be finite')
    if length <= experiment.evaluation_start + 1:
        raise ValueError('Sequence is shorter than the common warmup')
    experiment.model.eval()
    with torch.no_grad():
        # The injected fault is unknown at deployment, including in the PDF branch.
        predicted_a, predicted_pdf = experiment.model(a, u, g, torch.zeros_like(a))
    residual = (predicted_a - a[1:]).abs().reshape(-1)
    # PDF reconstruction is at the current time, not the next time.
    pdf_error = (predicted_pdf - pdf).square().mean(dim=1).sqrt()
    if not torch.isfinite(residual).all() or not torch.isfinite(pdf_error).all():
        raise ValueError('Model predictions must be finite')
    return predicted_a, predicted_pdf, residual, pdf_error


def _window_means(residual, start, window):
    # Each score includes the current target residual and only past observations.
    return torch.stack([residual[t-window+1:t+1].mean()
                        for t in range(start + window - 1, len(residual))])


def calibrate_normal(experiment, a, u, g, fault, pdf, k, window):
    if not isinstance(window, int) or window < 1 or k <= 0:
        raise ValueError('window_size must be positive integer and k must be positive')
    if torch.count_nonzero(fault):
        raise ValueError('Calibration must use independently confirmed normal data')
    _, _, residual, _ = _predict(experiment, a, u, g, fault, pdf)
    start = experiment.evaluation_start
    if len(residual) - start < window + 1:
        raise ValueError('Calibration requires at least two complete residual windows')
    scores = _window_means(residual, start, window)
    normalized = torch.stack([experiment._normalize_pdf(p) for p in pdf[start+1:]])
    return {
        'threshold': (scores.mean() + k * scores.std().clamp_min(1e-6)).item(),
        'pdf_mean': normalized.mean(dim=0),
        'pdf_std': normalized.std(dim=0).clamp_min(1e-3),
        'evaluation_start': start, 'window_size': window, 'k': k,
    }


def evaluate_without_labels(experiment, a, u, g, fault, pdf,
                            fault_start, fault_type, k, window, true_labels):
    calibration = experiment.normal_calibration
    if calibration is None:
        raise ValueError('Call calibrate_normal with independent normal data first')
    if (calibration['evaluation_start'], calibration['window_size'], calibration['k']) != (
            experiment.evaluation_start, window, k):
        raise ValueError('Calibration settings differ; recalibrate with the same settings')
    predicted_a, predicted_pdf, raw, pdf_error = _predict(experiment, a, u, g, fault, pdf)
    start = experiment.evaluation_start
    score_start = start + window - 1
    if score_start >= len(raw):
        raise ValueError('Test sequence has no complete evaluation window')
    experiment.window_size = window
    residuals = torch.full_like(raw, float('nan'))
    residuals[score_start:] = _window_means(raw, start, window)
    thresholds = torch.full_like(raw, calibration['threshold'])
    flags = torch.zeros_like(raw)
    flags[score_start:] = (residuals[score_start:] > thresholds[score_start:]).float()
    predicted_labels = torch.zeros(len(raw), dtype=torch.long, device=a.device)
    mean = calibration['pdf_mean'].to(pdf.device)
    std = calibration['pdf_std'].to(pdf.device)
    axis = torch.arange(pdf.shape[1], device=pdf.device, dtype=pdf.dtype) / max(1, pdf.shape[1]-1)
    diagnoses = []
    names = {0: '正常状态', 1: '药剂过多', 2: '药剂过少'}
    # Observed image PDFs are available online; fault identity/profile are not.
    for t in range(score_start, len(raw)):
        if flags[t] == 0:
            continue
        observed = experiment._normalize_pdf(pdf[t+1])
        features = experiment._extract_pdf_features(observed, (observed-mean)/std, axis)
        scores = experiment._compute_diagnosis_scores(features)
        code = max(scores, key=scores.get)
        predicted_labels[t] = code
        diagnoses.append((t, features['avg_size'], features['std_dev'],
                          features['skewness'], names[code], scores[code], code, features))

    # Ground truth enters only after all predictions have been computed.
    if true_labels is None:
        labels = np.zeros(len(a), dtype=int)
        if fault_type is not None:
            if fault_type not in (1, 2) or fault_start is None or not 0 <= fault_start < len(a):
                raise ValueError('Fault scoring requires a valid type and onset')
            labels[fault_start:] = fault_type
    else:
        labels = np.asarray(torch.as_tensor(true_labels).cpu()).reshape(-1)
        if len(labels) != len(a) or not np.isin(labels, [0, 1, 2]).all():
            raise ValueError('true_labels must contain one class 0/1/2 per observation')
    truth = labels[1:][score_start:]
    prediction = predicted_labels[score_start:].cpu().numpy()
    detection = flags[score_start:].cpu().numpy().astype(int)
    binary_truth = (truth != 0).astype(int)
    normal = binary_truth == 0
    faulty = ~normal
    target_times = np.arange(score_start+1, len(a))
    fault_times = target_times[faulty]
    onset = int(np.flatnonzero(labels != 0)[0]) if np.any(labels != 0) else None
    hits = target_times[(detection == 1) & faulty]
    delay = int(hits[0] - onset) if len(hits) else None
    return {
        'confusion_matrix': confusion_matrix(binary_truth, detection, labels=[0, 1]),
        'accuracy': accuracy_score(binary_truth, detection),
        'precision': precision_score(binary_truth, detection, zero_division=0),
        'recall': recall_score(binary_truth, detection, zero_division=0),
        'f1': f1_score(binary_truth, detection, zero_division=0),
        'diagnosis_accuracy': accuracy_score(truth, prediction),
        'macro_f1': f1_score(truth, prediction, labels=[0, 1, 2], average='macro', zero_division=0),
        'far': float(detection[normal].mean()) if normal.any() else None,
        'fdr': float(detection[faulty].mean()) if faulty.any() else None,
        'detection_delay': delay,
        'undetected': bool(len(fault_times) and not len(hits)),
        'onset_before_evaluation': bool(onset is not None and onset < score_start+1),
        'state_rmse': (predicted_a[start:] - a[1:][start:]).square().mean().sqrt().item(),
        'pdf_rmse': pdf_error[start+1:].square().mean().sqrt().item(),
        'pred_a_seq': predicted_a, 'pred_pdf_seq': predicted_pdf,
        'fault_flags': flags, 'residuals': residuals, 'thresholds': thresholds,
        'predicted_labels': predicted_labels, 'diagnosis_results': diagnoses,
        'evaluation_mask': torch.arange(len(raw), device=a.device) >= score_start,
        'evaluation_start': start, 'score_start': score_start,
        'feature_trends': experiment._compute_feature_trends(pdf, mean, std),
    }
