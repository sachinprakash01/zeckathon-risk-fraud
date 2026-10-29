"""End-to-end pipeline:  python run_pipeline.py --data data/raw/<paysim>.csv

load -> causal features -> chronological split -> CatBoost (risky types only)
-> thresholds fitted on VALIDATION -> frozen on TEST -> rules -> SHAP reasons -> artifacts/
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data as D
from src import evaluate as E
from src import features as F
from src import model as M
from src import rules as R


def main(a):
    os.makedirs(a.out, exist_ok=True)
    print("[1/7] loading", a.data)
    df = D.load(a.data)
    print(f"      {len(df):,} rows, fraud rate {df.isFraud.mean():.4%}")

    print("[2/7] causal feature engineering")
    df = F.add_features(df)

    print("[3/7] chronological split")
    train, valid, test, cuts = D.time_split(df)
    for n, d in [("train", train), ("valid", valid), ("test", test)]:
        print(f"      {n}: {len(d):,} rows, {int(d.isFraud.sum()):,} fraud")
    if min(valid.isFraud.sum(), test.isFraud.sum()) < 20:
        print("      WARNING: very few fraud cases in valid/test; metrics will be noisy.")

    print("[4/7] training CatBoost on TRANSFER/CASH_OUT rows")
    tr_r, va_r = train[F.risky_mask(train)], valid[F.risky_mask(valid)]
    model = M.train(tr_r, va_r, seed=a.seed)
    model.save_model(os.path.join(a.out, "model.cbm"))

    def full_scores(d):
        s = np.zeros(len(d))
        m = F.risky_mask(d).to_numpy()
        if m.any():
            s[m] = M.score(model, d[m])
        return s

    print("[5/7] fitting thresholds on VALIDATION, freezing for TEST")
    s_va, s_te = full_scores(valid), full_scores(test)
    if a.mode == "target":
        thr = R.fit_thresholds_target(s_va, valid["isFraud"].to_numpy(), a.flag_precision, a.review_recall)
    else:
        thr = R.fit_thresholds(s_va, a.flag_budget, a.review_budget)
    print(f"      mode={a.mode}  flag>={thr['flag']:.4f}  review>={thr['review']:.4f}")
    json.dump(thr, open(os.path.join(a.out, "thresholds.json"), "w"), indent=2)

    y = test["isFraud"].to_numpy()
    risky_te = F.risky_mask(test).to_numpy()
    base_tier = R.model_tier(s_te, thr)
    rules_tier, fired = R.apply_rules(test, base_tier, risky_te)
    # Rules are an ablation by default: on PaySim they add alerts without adding recall.
    final_tier = rules_tier if a.use_rules else base_tier

    print("[6/7] evaluating")
    # naive rule-only baseline: drain rule alone
    rule_only = ((test["drains_origin"].to_numpy() == 1) & risky_te).astype(float)
    rec_b = {f"{b:.2%}": dict(zip(("recall", "precision"), E.recall_at_budget(y, s_te, b)))
             for b in (0.001, 0.005, 0.01, 0.0134, 0.02)}
    metrics = {
        "cuts": cuts,
        "model": E.summary(y, s_te),
        "rule_only_baseline": E.summary(y, rule_only),
        "recall_at_review_budget": rec_b,
        "tiers_model_only": E.tier_table(y, base_tier),
        "tiers_model_plus_rules": E.tier_table(y, rules_tier),
        "tiers_final": E.tier_table(y, final_tier),
        "rules_used_in_final": bool(a.use_rules),
        "thresholds": thr,
    }
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)

    print("[7/7] SHAP reasons + plots")
    keep = (final_tier >= 1) | (y == 1)
    idx = np.where(keep)[0]
    if len(idx) > a.max_cases:  # keep all fraud + top-scored others
        fraud_idx = idx[y[idx] == 1]
        others = idx[y[idx] == 0]
        others = others[np.argsort(-s_te[others])[: max(a.max_cases - len(fraud_idx), 0)]]
        idx = np.sort(np.concatenate([fraud_idx, others]))
    cases = test.iloc[idx].copy()
    cases["score"] = s_te[idx]
    cases["model_tier"] = [R.TIER_NAME[t] for t in base_tier[idx]]
    cases["decision"] = [R.TIER_NAME[t] for t in final_tier[idx]]
    cases["rule_fired"] = [fired[i] for i in idx]
    risky_cases = F.risky_mask(cases).to_numpy()
    sv = np.zeros((len(cases), len(F.FEATURES) + 1))
    if risky_cases.any():
        sv[risky_cases] = M.shap_values(model, cases[risky_cases])
    for j, f in enumerate(F.FEATURES):
        cases[f"shap_{f}"] = sv[:, j]
    cases["reasons"] = [
        "; ".join(R.build_reasons(cases.iloc[i], sv[i], F.FEATURES)) if risky_cases[i] else ""
        for i in range(len(cases))]
    cases["reasons"] = np.where((cases["reasons"] == "") & (cases["rule_fired"] != ""),
                                "rule: " + cases["rule_fired"], cases["reasons"])
    cases.drop(columns=["isFlaggedFraud"], errors="ignore").to_csv(
        os.path.join(a.out, "cases.csv"), index=False)

    # global SHAP importance on a sample of risky test rows
    samp = test[risky_te]
    samp = samp.sample(min(len(samp), 20000), random_state=0)
    gsv = M.shap_values(model, samp)[:, :-1]
    imp = pd.Series(np.abs(gsv).mean(0), index=F.FEATURES).sort_values()
    imp.to_csv(os.path.join(a.out, "global_shap.csv"), header=["mean_abs_shap"])
    plt.figure(figsize=(7, 5)); imp.plot.barh(color="#2a6fdb")
    plt.title("Global feature impact (mean |SHAP|, test sample)"); plt.tight_layout()
    plt.savefig(os.path.join(a.out, "global_shap.png"), dpi=140); plt.close()

    p, r = E.pr_curve(y, s_te)
    plt.figure(figsize=(6, 4.5)); plt.plot(r, p, label=f"CatBoost (PR-AUC {metrics['model']['pr_auc']:.3f})")
    plt.axhline(metrics["model"]["base_rate"], ls="--", c="gray", label="random")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("Precision-Recall, TEST period")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "pr_curve.png"), dpi=140); plt.close()

    print("\n=== TEST RESULTS ===")
    print(f"PR-AUC model: {metrics['model']['pr_auc']:.4f} | drain-rule baseline: {metrics['rule_only_baseline']['pr_auc']:.4f}")
    for k, v in rec_b.items():
        print(f"review top {k}: recall {v['recall']:.1%}, precision {v['precision']:.1%}")
    for name in ("tiers_model_only", "tiers_model_plus_rules", "tiers_final"):
        t = metrics[name]["FLAG+REVIEW"]
        print(f"{name}: FLAG+REVIEW = {t['transactions']:,} txns, recall {t['recall']:.1%}, precision {t['precision']:.1%}")
    print("artifacts written to", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--use-rules", action="store_true", dest="use_rules",
                    help="escalate decisions with R1/R2 (off by default; see README ablation)")
    ap.add_argument("--mode", choices=["target", "budget"], default="target")
    ap.add_argument("--flag-precision", type=float, default=0.999, dest="flag_precision")
    ap.add_argument("--review-recall", type=float, default=0.99, dest="review_recall")
    ap.add_argument("--flag-budget", type=float, default=0.001, dest="flag_budget")
    ap.add_argument("--review-budget", type=float, default=0.003, dest="review_budget")
    ap.add_argument("--max-cases", type=int, default=20000, dest="max_cases")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
