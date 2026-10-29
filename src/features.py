"""Feature engineering - leakage-aware.

Design rules (documented in README):
1. Only information available *at decision time* is used.
   - We use PRE-transaction balances (oldbalance*) and the amount.
   - We DROP newbalanceOrig / newbalanceDest: they are computed after the
     transaction and in PaySim they encode the fraud mechanics (label leakage).
   - We DROP isFlaggedFraud (a crude built-in rule) and the raw account-name strings.
2. History features are strictly causal: counts only include EARLIER rows,
   never the current/future ones.
3. Fraud in PaySim only occurs in TRANSFER and CASH_OUT, so the model scores
   only those types; everything else is auto-cleared by policy.
"""
import numpy as np
import pandas as pd

RISKY_TYPES = ["TRANSFER", "CASH_OUT"]

FEATURES = [
    "type", "amount", "log_amount", "oldbalanceOrg", "oldbalanceDest",
    "amount_to_orig_balance", "drains_origin", "amount_exceeds_balance", "orig_balance_zero",
    "dest_balance_zero", "dest_is_merchant", "dest_prior_txn_count",
    "dest_first_seen", "hour_of_day",
]
CAT_FEATURES = ["type"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("step", kind="stable").copy()
    eps = 1.0
    df["log_amount"] = np.log1p(df["amount"])
    df["amount_to_orig_balance"] = df["amount"] / (df["oldbalanceOrg"] + eps)
    # exact drain: amount is within 2% of the sender's balance (NOT merely larger than it:
    # PaySim does not enforce balances, so amount > balance is common and mostly benign)
    df["drains_origin"] = ((df["oldbalanceOrg"] > 0) &
                           ((df["amount"] - df["oldbalanceOrg"]).abs() <= 0.02 * df["oldbalanceOrg"])).astype(int)
    df["amount_exceeds_balance"] = (df["amount"] > 1.02 * df["oldbalanceOrg"]).astype(int)
    df["orig_balance_zero"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["dest_balance_zero"] = (df["oldbalanceDest"] == 0).astype(int)
    df["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
    df["dest_prior_txn_count"] = df.groupby("nameDest").cumcount()
    df["dest_first_seen"] = (df["dest_prior_txn_count"] == 0).astype(int)
    df["hour_of_day"] = df["step"] % 24
    df["day"] = (df["step"] - 1) // 24
    return df


def risky_mask(df: pd.DataFrame) -> pd.Series:
    return df["type"].isin(RISKY_TYPES)
