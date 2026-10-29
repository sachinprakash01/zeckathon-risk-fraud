"""Loading and time-based splitting.

Splits are strictly chronological on the `step` column (1 step = 1 hour in PaySim).
No random shuffling anywhere: the model never trains on the future.
"""
import pandas as pd

REQUIRED = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
            "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Is this the PaySim file?")
    return df.sort_values("step", kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, train_frac=0.65, valid_frac=0.15):
    """Chronological train / valid / test split by `step` cutoffs."""
    max_step = df["step"].max()
    t_cut = int(max_step * train_frac)
    v_cut = int(max_step * (train_frac + valid_frac))
    train = df[df["step"] <= t_cut]
    valid = df[(df["step"] > t_cut) & (df["step"] <= v_cut)]
    test = df[df["step"] > v_cut]
    cuts = {"train_max_step": t_cut, "valid_max_step": v_cut, "max_step": int(max_step)}
    return train, valid, test, cuts
