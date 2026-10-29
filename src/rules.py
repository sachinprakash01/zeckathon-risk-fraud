"""Decision layer: model score + transparent rules -> FLAG / REVIEW / CLEAR.

Tiering is driven by an analyst-capacity budget, not an arbitrary 0.5 cut-off:
  * FLAG   = top `flag_budget` share of all transactions by score  (auto-hold / call customer)
  * REVIEW = next `review_budget` share                              (analyst queue)
  * CLEAR  = everything else
Thresholds are fitted on the VALIDATION period only and then frozen for test.

Two explicit, human-readable rules can only ESCALATE a decision (never downgrade):
  R1  account drain on a TRANSFER / CASH_OUT            -> at least REVIEW
  R2  drain + empty destination + first-time destination -> FLAG
Every decision carries plain-English reasons derived from per-row SHAP values.
"""
import numpy as np
import pandas as pd

TIER_ORDER = {"CLEAR": 0, "REVIEW": 1, "FLAG": 2}
TIER_NAME = {v: k for k, v in TIER_ORDER.items()}


def fit_thresholds(valid_scores, flag_budget=0.001, review_budget=0.003):
    s = np.sort(np.asarray(valid_scores))[::-1]
    n = len(s)
    k_flag = max(int(round(n * flag_budget)), 1)
    k_rev = max(int(round(n * (flag_budget + review_budget))), k_flag + 1)
    return {"flag": float(s[min(k_flag, n) - 1]), "review": float(s[min(k_rev, n) - 1]),
            "flag_budget": flag_budget, "review_budget": review_budget}


def fit_thresholds_target(valid_scores, y_valid, flag_precision=0.95, review_recall=0.98):
    """Target-based thresholds fitted on VALIDATION only.
    FLAG   = lowest score cut that still keeps precision >= flag_precision
    REVIEW = lowest score cut that reaches recall >= review_recall
    Better than a fixed capacity budget when fraud is highly separable."""
    s = np.asarray(valid_scores, dtype=float)
    y = np.asarray(y_valid, dtype=int)
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(s) + 1)
    precision = tp / k
    recall = tp / max(y.sum(), 1)
    ok = np.where(precision >= flag_precision)[0]
    k_flag = int(ok.max()) + 1 if len(ok) else 1
    hit = np.where(recall >= review_recall)[0]
    k_rev = int(hit.min()) + 1 if len(hit) else len(s)
    k_rev = max(k_rev, k_flag)
    return {"flag": float(s_sorted[k_flag - 1]), "review": float(s_sorted[k_rev - 1]),
            "mode": "target", "flag_precision": flag_precision, "review_recall": review_recall}


def model_tier(scores, thr):
    s = np.asarray(scores)
    tier = np.zeros(len(s), dtype=int)
    tier[s >= thr["review"]] = 1
    tier[s >= thr["flag"]] = 2
    return tier


def apply_rules(df, base_tier, risky):
    """Return (final_tier, rule_ids) where rule_ids is a list of fired rule names per row."""
    tier = base_tier.copy()
    risky = np.asarray(risky, dtype=bool)
    r1 = risky & (df["drains_origin"].to_numpy() == 1)
    r2 = r1 & (df["dest_balance_zero"].to_numpy() == 1) & (df["dest_first_seen"].to_numpy() == 1)
    tier = np.where(r1, np.maximum(tier, 1), tier)
    tier = np.where(r2, np.maximum(tier, 2), tier)
    fired = [("R2_drain_to_empty_new_dest" if b else "R1_account_drain" if a else "")
             for a, b in zip(r1, r2)]
    return tier, fired


def _pct(x):
    return f"{x:.0%}" if x < 10 else f"{x:.0f}x"


# binary features: (text when value is 1, text when value is 0) so the wording always matches the data
BINARY = {
    "drains_origin": ("the sender's account is emptied by this transaction",
                      "the transaction does not empty the sender's account"),
    "amount_exceeds_balance": ("amount is larger than the sender's available balance",
                               "amount does not exceed the sender's balance"),
    "orig_balance_zero": ("sender balance was already zero", "sender had a non-zero balance"),
    "dest_balance_zero": ("destination account held a zero balance before",
                          "destination account already held a balance"),
    "dest_first_seen": ("destination account has never received money before",
                        "destination account has received money before"),
    "dest_is_merchant": ("destination is a merchant account", "destination is a personal account"),
}

TEMPLATES = {
    "amount_to_orig_balance": lambda r: f"amount is {_pct(r['amount_to_orig_balance'])} of the sender's balance",
    "amount": lambda r: f"transaction amount is {r['amount']:,.0f}",
    "log_amount": lambda r: f"transaction amount is {r['amount']:,.0f}",
    "oldbalanceOrg": lambda r: f"sender balance before the transaction was {r['oldbalanceOrg']:,.0f}",
    "oldbalanceDest": lambda r: f"destination balance before was {r['oldbalanceDest']:,.0f}",
    "dest_prior_txn_count": lambda r: f"destination has only {int(r['dest_prior_txn_count'])} earlier incoming transactions",
    "type": lambda r: f"transaction type is {r['type']}",
    "hour_of_day": lambda r: f"occurs at hour {int(r['hour_of_day'])} of the day",
    "day": lambda r: f"occurs on simulation day {int(r['day'])}",
}
for _f, (_t1, _t0) in BINARY.items():
    TEMPLATES[_f] = (lambda t1, t0, f: (lambda r: t1 if r[f] == 1 else t0))(_t1, _t0, _f)


def build_reasons(row, shap_row, feature_names, top_k=3, min_contrib=0.05):
    """Top positive SHAP drivers rendered as plain English (de-duplicated)."""
    contrib = pd.Series(shap_row[:-1], index=feature_names).sort_values(ascending=False)
    reasons, seen = [], set()
    for feat, val in contrib.items():
        if val < min_contrib or len(reasons) >= top_k:
            break
        if feat == "amount_to_orig_balance" and row["drains_origin"] == 1:
            continue  # already covered by the "account is emptied" reason
        text = TEMPLATES[feat](row)
        if text not in seen:
            seen.add(text)
            reasons.append(text)
    return reasons
