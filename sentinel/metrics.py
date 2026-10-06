"""Classification metrics implemented from scratch (no sklearn required)."""
from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def roc_auc(y_true, scores):
    """AUC via the Mann-Whitney U statistic (handles ties)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    order = scores.argsort()
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y_true, scores, steps=200):
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    fpr, tpr = [], []
    for t in np.linspace(1, 0, steps):
        tp, tn, fp, fn = confusion(y_true, (scores >= t).astype(int))
        tpr.append(tp / max(tp + fn, 1))
        fpr.append(fp / max(fp + tn, 1))
    return np.array(fpr), np.array(tpr)


def report(y_true, scores, threshold=0.5):
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tp, tn, fp, fn = confusion(y_true, y_pred)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "roc_auc": roc_auc(y_true, scores),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
