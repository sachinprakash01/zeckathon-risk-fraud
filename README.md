# Explainable Fraud Triage (Zeckathon 2026 - Risk & Fraud Analytics)

`artifacts/` already contains the results of the run described below, so the app can be viewed without PaySim: `pip install -r requirements.txt`, then `streamlit run app.py`.

A payment-fraud triage system that turns a CatBoost risk score into a **FLAG / REVIEW / CLEAR**
decision, with plain-English reasons for every case and a fixed analyst-capacity budget.

> Results below come from the real PaySim file (Lopez-Rojas et al., Kaggle: *Synthetic Financial Datasets For Fraud Detection*).
> PaySim is itself synthetic: see Limitations.

## Problem
Fraud teams cannot review every transaction. The real question is not "what is the accuracy?"
(a model that never predicts fraud is ~99.9% accurate on PaySim) but:
**"If analysts can review only the top X% of transactions, how much fraud do we catch, and can we explain each alert?"**

## Approach
1. **Leakage-aware features** - only information available at decision time.
2. **Chronological split** - train < valid < test by `step`; no shuffling.
3. **CatBoost** trained on TRANSFER / CASH_OUT rows (the only types containing fraud in PaySim); other types auto-clear by policy.
4. **Target-based thresholds** - FLAG = lowest score keeping validation precision >= 99.9%; REVIEW = lowest score reaching >= 99% recall. Both are fitted on the *validation* period and frozen for test. (A fixed capacity-budget mode is available with `--mode budget`.)
5. **Transparent rule layer, kept as an ablation (off by default; `--use-rules` enables it).** R1 account drain -> REVIEW; R2 drain + empty + first-time destination -> FLAG. On PaySim the rules add alerts without adding recall, so the final system is model-only.
6. **SHAP reasons** - per-transaction TreeSHAP drivers rendered as English sentences.
7. **Analyst console** (Streamlit): review queue, case explainer, and a tab of *missed* fraud.

## Leakage controls (what we dropped and why)
| Column | Decision | Reason |
|---|---|---|
| `newbalanceOrig`, `newbalanceDest` | dropped | computed after the event; encode the fraud mechanics |
| `isFlaggedFraud` | dropped | crude built-in rule, not available as an independent signal |
| `nameOrig`, `nameDest` strings | dropped | high-cardinality IDs; only causal history from `nameDest` is used |
| `dest_prior_txn_count` | kept | counts only *earlier* rows (see `tests/test_pipeline.py`) |
| `day` (absolute simulation day) | dropped | ranked 4th in SHAP importance; absolute time is not a real fraud signal and the test period lies outside the training days. Removing it did not change test PR-AUC |
| "drain" definition | amount within 2% of balance | amount > balance is common in PaySim and mostly benign; a first version using `>=` fired on thousands of benign rows and was fixed |

## Run it
```bash
pip install -r requirements.txt
# download PaySim from Kaggle (Online Payments Fraud Detection / PaySim) into data/raw/
python run_pipeline.py --data data/raw/<paysim_file>.csv
python -m pytest -q tests
streamlit run app.py
```
Options: `--flag-precision 0.999 --review-recall 0.99 --use-rules --mode budget --flag-budget 0.001 --review-budget 0.003 --seed 42`.
`python -m src.make_synthetic` creates a small fake dataset **for smoke-testing the code only**.

## Results (chronological hold-out test period)
Test period: 123,580 transactions, 1,654 fraud (1.34%; the full dataset is 0.13%).

| System | Alerts | Recall | Precision |
|---|---|---|---|
| **Final: CatBoost model only** | 1,654 | 99.9% | 99.9% |
| Ablation: model + rules R1/R2 | 1,837 | 99.9% | 90.0% |

- PR-AUC (test): ~1.000. Naive "account drain" rule alone: 0.866.
- Thresholds were fitted on the validation period and frozen for test (no test-set tuning).
- Rules did not help on PaySim: +183 alerts, no recall gain, precision down from 99.9% to 90.0%.
- FLAG and REVIEW thresholds coincide on PaySim because fraud is almost perfectly separable, so there is effectively one alert tier. On noisier real-world data the tiers would separate.
- Top SHAP drivers: sender's pre-transaction balance, amount-to-balance ratio, and "account drained" - the scripted PaySim fraud pattern, not a label leak.
- Full numbers: `artifacts/metrics.json`.

## Limitations (honest)
- PaySim is **synthetic** with a scripted fraud pattern (account drain). Near-perfect scores reflect the simulator, not expected real-world performance. Treat metrics as a method demonstration.
- The test period has a higher fraud rate (1.34%) than the full dataset (0.13%).
- Fraud only appears in TRANSFER/CASH_OUT, so the "auto-clear other types" policy is dataset-specific.
- Rules lowered precision without raising recall here; we report both model-only and model+rules so the ablation is visible.
- No concept-drift or fairness analysis is included.

## AI-tool usage disclosure
This project's initial code scaffold (pipeline, rule layer, Streamlit app, tests) was generated with **Claude (Anthropic)**
and then run, tested and reviewed by the author. Design choices (dataset, leakage rules, budget-based thresholds)
were discussed with Claude; the author is responsible for the final submission.
