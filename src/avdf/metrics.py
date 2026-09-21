import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def eer(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
    return xs, ys, ns


def classification_metrics(y, p_fake, conf=None, threshold=0.5):
    y, p_fake = np.asarray(y), np.asarray(p_fake)
    pred = (p_fake >= threshold).astype(int)
    conf = np.maximum(p_fake, 1 - p_fake) if conf is None else np.asarray(conf)
    out = dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
               ece=ece(conf, (pred == y).astype(float)))
    if len(np.unique(y)) == 2:
        out["auc"] = roc_auc_score(y, p_fake)
        out["eer"] = eer(y, p_fake)
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def selective_metrics(y, pred, review):
    y, pred, review = map(np.asarray, (y, pred, review))
    acc_mask = ~review
    wrong = pred != y
    return dict(coverage=float(acc_mask.mean()), flagged_pct=float(review.mean() * 100),
                selective_accuracy=float((pred[acc_mask] == y[acc_mask]).mean()) if acc_mask.any() else float("nan"),
                errors_caught=float(review[wrong].mean()) if wrong.any() else float("nan"))
