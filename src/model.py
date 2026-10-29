"""CatBoost training. Negatives in the TRAIN set are down-sampled for speed;
validation and test are never resampled, so metrics reflect the real class balance."""
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .features import FEATURES, CAT_FEATURES


def downsample_negatives(df, ratio=50, seed=42):
    pos = df[df["isFraud"] == 1]
    neg = df[df["isFraud"] == 0]
    n_neg = min(len(neg), max(ratio * len(pos), 1000))
    neg = neg.sample(n=n_neg, random_state=seed)
    return pd.concat([pos, neg]).sort_values("step")


def train(train_df, valid_df, seed=42, iterations=600):
    tr = downsample_negatives(train_df, seed=seed)
    train_pool = Pool(tr[FEATURES], tr["isFraud"], cat_features=CAT_FEATURES)
    valid_pool = Pool(valid_df[FEATURES], valid_df["isFraud"], cat_features=CAT_FEATURES)
    model = CatBoostClassifier(
        iterations=iterations, learning_rate=0.05, depth=6, eval_metric="PRAUC",
        loss_function="Logloss", random_seed=seed, verbose=100,
        od_type="Iter", od_wait=60, use_best_model=True,
    )
    model.fit(train_pool, eval_set=valid_pool)
    return model


def score(model, df):
    pool = Pool(df[FEATURES], cat_features=CAT_FEATURES)
    return model.predict_proba(pool)[:, 1]


def shap_values(model, df):
    """Per-row TreeSHAP contributions (log-odds). Last column is the base value."""
    pool = Pool(df[FEATURES], cat_features=CAT_FEATURES)
    return model.get_feature_importance(pool, type="ShapValues")
