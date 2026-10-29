import numpy as np
import pandas as pd

from src import data as D, features as F
from src.make_synthetic import make


def _df():
    return make(n=20000, seed=1)


def test_split_is_chronological_and_disjoint():
    tr, va, te, _ = D.time_split(_df())
    assert tr["step"].max() < va["step"].min() <= va["step"].max() < te["step"].min()
    assert len(tr) + len(va) + len(te) == 20000


def test_no_post_transaction_columns_used():
    assert not any(c.startswith("newbalance") for c in F.FEATURES)
    assert "isFlaggedFraud" not in F.FEATURES and "isFraud" not in F.FEATURES


def test_history_feature_is_causal():
    df = F.add_features(_df())
    first = df.groupby("nameDest").head(1)
    assert (first["dest_prior_txn_count"] == 0).all()   # first sighting never counts itself
    assert df["dest_prior_txn_count"].max() < len(df)


def test_drain_definition_not_triggered_by_amount_above_balance():
    d = pd.DataFrame({"step": [1], "type": ["TRANSFER"], "amount": [5000.0], "nameOrig": ["C1"],
                      "oldbalanceOrg": [100.0], "newbalanceOrig": [0.0], "nameDest": ["C2"],
                      "oldbalanceDest": [0.0], "newbalanceDest": [5000.0], "isFraud": [0]})
    f = F.add_features(d)
    assert f.loc[0, "drains_origin"] == 0 and f.loc[0, "amount_exceeds_balance"] == 1
