"""Competition-aligned metrics and threshold optimization for model v3."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


LABELS = np.array([0, 1, 2], dtype=np.int8)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         proba: np.ndarray | None = None) -> dict:
    """Return all official score components plus precision/recall/F1/AUC."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    yt = y_true != 0
    yp = y_pred != 0
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    fn = int(np.sum(yt & ~yp))
    threat_precision = tp / (tp + fp) if tp + fp else 0.0
    threat_recall = tp / (tp + fn) if tp + fn else 0.0
    threat_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    p = precision_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    soft = np.where(y_true == y_pred, 1.0,
                    np.where((y_true != 0) & (y_pred != 0), 0.5, 0.0)).mean()
    threat_class_recall = float((r[1] + r[2]) / 2)
    macro_f1 = float(f.mean())
    balanced_accuracy = float(r.mean())
    final_score = (
        0.40 * threat_f1
        + 0.25 * threat_recall
        + 0.15 * threat_class_recall
        + 0.10 * macro_f1
        + 0.05 * float(soft)
        + 0.05 * balanced_accuracy
    )
    result = {
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": macro_f1,
        "precision_benign": float(p[0]),
        "precision_suspicious": float(p[1]),
        "precision_malicious": float(p[2]),
        "recall_benign": float(r[0]),
        "recall_suspicious": float(r[1]),
        "recall_malicious": float(r[2]),
        "f1_benign": float(f[0]),
        "f1_suspicious": float(f[1]),
        "f1_malicious": float(f[2]),
        "threat_precision": float(threat_precision),
        "threat_recall": float(threat_recall),
        "threat_f1": float(threat_f1),
        "threat_class_recall": threat_class_recall,
        "soft_label_score": float(soft),
        "balanced_accuracy": balanced_accuracy,
        "competition_score": float(final_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "false_positives": fp,
        "false_negatives": fn,
    }
    if proba is not None:
        try:
            result["auc_ovr_macro"] = float(
                roc_auc_score(y_true, proba, labels=LABELS, multi_class="ovr", average="macro")
            )
            result["auc_threat"] = float(roc_auc_score(yt.astype(np.int8), 1.0 - proba[:, 0]))
        except ValueError:
            result["auc_ovr_macro"] = None
            result["auc_threat"] = None
    return result


def combine_probabilities(multiclass_proba: np.ndarray, threat_proba: np.ndarray,
                          subtype_proba: np.ndarray, threat_weight: float,
                          subtype_weight: float) -> np.ndarray:
    """Blend hierarchical and multiclass probabilities without label leakage."""
    multi_threat = 1.0 - multiclass_proba[:, 0]
    p_threat = threat_weight * threat_proba + (1.0 - threat_weight) * multi_threat
    multi_subtype = multiclass_proba[:, 2] / np.maximum(
        multiclass_proba[:, 1] + multiclass_proba[:, 2], 1e-9
    )
    p_mal_given_threat = subtype_weight * subtype_proba + (1.0 - subtype_weight) * multi_subtype
    result = np.empty_like(multiclass_proba, dtype=np.float64)
    result[:, 0] = 1.0 - p_threat
    result[:, 2] = p_threat * p_mal_given_threat
    result[:, 1] = p_threat * (1.0 - p_mal_given_threat)
    return result


def threshold_predict(proba: np.ndarray, threat_threshold: float,
                      malicious_threshold: float) -> np.ndarray:
    p_threat = 1.0 - proba[:, 0]
    p_mal_given_threat = proba[:, 2] / np.maximum(p_threat, 1e-9)
    pred = np.zeros(len(proba), dtype=np.int8)
    threat = p_threat >= threat_threshold
    pred[threat] = np.where(p_mal_given_threat[threat] >= malicious_threshold, 2, 1)
    return pred


def optimize_thresholds(y_true: np.ndarray, multiclass_proba: np.ndarray,
                        threat_proba: np.ndarray, subtype_proba: np.ndarray,
                        config: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """Optimize the complete official score subject to recall safeguards."""
    best = None
    best_pred = None
    best_proba = None
    fallback = None
    for tw in config["ensemble_weights"]:
        for sw in config["ensemble_weights"]:
            proba = combine_probabilities(multiclass_proba, threat_proba, subtype_proba, tw, sw)
            for tt in config["threat_thresholds"]:
                for mt in config["malicious_thresholds"]:
                    pred = threshold_predict(proba, tt, mt)
                    metrics = evaluate_predictions(y_true, pred, proba)
                    candidate = {
                        "threat_weight": float(tw),
                        "subtype_weight": float(sw),
                        "threat_threshold": float(tt),
                        "malicious_threshold": float(mt),
                        "metrics": metrics,
                    }
                    key = (metrics["competition_score"], metrics["threat_f1"])
                    if fallback is None or key > fallback[0]:
                        fallback = (key, candidate, pred.copy(), proba.copy())
                    feasible = (
                        metrics["recall_malicious"] >= config["minimum_malicious_recall"]
                        and metrics["threat_recall"] >= config["minimum_threat_recall"]
                    )
                    if feasible and (best is None or key > best[0]):
                        best = (key, candidate, pred.copy(), proba.copy())
    selected = best if best is not None else fallback
    assert selected is not None
    selected[1]["recall_constraints_satisfied"] = best is not None
    return selected[1], selected[2], selected[3]
