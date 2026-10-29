"""Generate a small PaySim-style dataset for smoke-testing the pipeline.

NOT used for reported results. Real results come from the Kaggle PaySim file.
Schema matches PaySim: step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,
nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud
"""
import numpy as np
import pandas as pd


def make(n=300_000, steps=743, seed=0):
    rng = np.random.default_rng(seed)
    types = rng.choice(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"], size=n,
                       p=[0.34, 0.22, 0.35, 0.08, 0.01])
    step = np.sort(rng.integers(1, steps + 1, size=n))
    amount = np.round(rng.lognormal(10, 1.4, size=n), 2)
    old_o = np.round(rng.lognormal(10.5, 2.0, size=n), 2) * (rng.random(n) > 0.25)
    orig = np.array([f"C{i}" for i in rng.integers(0, n, size=n)])
    is_merchant = types == "PAYMENT"
    dest_ids = rng.integers(0, n // 3, size=n)
    dest = np.where(is_merchant, [f"M{i}" for i in dest_ids], [f"C{i}" for i in dest_ids])
    old_d = np.round(rng.lognormal(11, 2.0, size=n), 2) * (rng.random(n) > 0.35)

    risky = np.isin(types, ["TRANSFER", "CASH_OUT"])
    fraud = risky & (rng.random(n) < 0.015)
    drain = fraud & (rng.random(n) < 0.8)
    amount = np.where(drain, old_o, amount)
    amount = np.where(fraud & ~drain, np.minimum(amount * 3, np.maximum(old_o, 1)), amount)
    old_d = np.where(fraud & (rng.random(n) < 0.6), 0.0, old_d)
    legit_drain = risky & ~fraud & (rng.random(n) < 0.01)
    amount = np.where(legit_drain, old_o, amount)

    new_o = np.maximum(old_o - amount, 0)
    new_d = np.where(is_merchant, 0.0, old_d + amount)
    return pd.DataFrame(dict(step=step, type=types, amount=amount, nameOrig=orig,
                             oldbalanceOrg=old_o, newbalanceOrig=new_o, nameDest=dest,
                             oldbalanceDest=old_d, newbalanceDest=new_d,
                             isFraud=fraud.astype(int), isFlaggedFraud=0))


if __name__ == "__main__":
    df = make()
    df.to_csv("data/raw/synthetic_paysim.csv", index=False)
    print(df.shape, "fraud rate:", round(df.isFraud.mean(), 5))
