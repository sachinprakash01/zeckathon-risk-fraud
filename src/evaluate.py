"""Evaluation helpers. We report metrics that matter for a review-capacity problem,
not accuracy (a model that says 'never fraud' is 99.9% accurate on PaySim)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


def recall_at_budget(y, scores, budget_frac):
    """Share of all fraud caught if analysts can review the top `budget_frac` of transactions."""
    y = np.asarray(y); s = np.asarray(scores)
    k = max(int(round(len(y) * budget_frac)), 1)
    idx = np.argsort(-s, kind="stable")[:k]
    return float(y[idx].sum() / max(y.sum(), 1)), float(y[idx].mean())


def tier_table(y, tier):
    y = np.asarray(y); tier = np.asarray(tier)
    out = {}
    total_fraud = int(y.sum())
    for t, name in [(2, "FLAG"), (1, "REVIEW"), (0, "CLEAR")]:
        m = tier == t
        out[name] = {"transactions": int(m.sum()), "fraud": int(y[m].sum()),
                     "precision": float(y[m].mean()) if m.any() else 0.0,
                     "share_of_all_fraud": float(y[m].sum() / max(total_fraud, 1))}
    caught = tier >= 1
    out["FLAG+REVIEW"] = {"transactions": int(caught.sum()), "fraud": int(y[caught].sum()),
                          "precision": float(y[caught].mean()) if caught.any() else 0.0,
                          "recall": float(y[caught].sum() / max(total_fraud, 1))}
    return out


def summary(y, scores):
    y = np.asarray(y)
    return {"pr_auc": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)) if y.min() != y.max() else None,
            "base_rate": float(y.mean()), "n": int(len(y)), "n_fraud": int(y.sum())}


def pr_curve(y, scores):
    p, r, _ = precision_recall_curve(y, scores)
    return p, r
