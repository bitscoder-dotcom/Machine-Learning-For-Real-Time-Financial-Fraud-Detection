# Phase 2 Summary — Data Understanding + Leakage-Proof Splitting (Time-Aware)

## Goal
Extend Phase 1 by adding **time-aware evaluation** and **leakage-proof controls**, while keeping runs reproducible (saved splits + logged artifacts).

## Datasets (what we’re modelling)

### 1) creditcard.csv
- **Label:** `Class` (fraud=1)
- **Order field:** `Time` (seconds since first transaction; an ordering proxy)
- **Features used (baseline pipeline):** numeric only (`Time`, `Amount`, `V1–V28`)
- **Split used in Phase 2 runs:** time-aware 70/15/15 based on `Time`  
  Saved split: `results/splits/creditcard_time_Time_70_15_15.npz`

### 2) IEEE-CIS (train_transaction.csv)
- **Label:** `isFraud`
- **Order field:** `TransactionDT`
- **Identifier handling:** `TransactionID` treated as an ID and excluded from features
- **Features used (baseline pipeline):** numeric only (categoricals not encoded yet)
- **Split used in Phase 2 runs:** time-aware 70/15/15 based on `TransactionDT`  
  Saved split: `results/splits/ieee_time_TransactionDT_70_15_15.npz`

---

## Leakage-proof controls added (what’s enforced)
- **Time-aware split option** to prevent training on later transactions and testing on earlier ones.
- **IDs excluded from features** (e.g., `TransactionID`).
- **Train-only preprocessing**: scalers/encoders are fitted on `X_train` only (via sklearn Pipeline), then applied to val/test.
- **Reproducibility**: split indices saved as `.npz`, and each run logs metrics + model/metrics paths into `results/metrics/runs.csv`.

---

## Phase 2 leaderboard (from `runs.csv`)
Sort key: `test_pr_auc`

### A) creditcard.csv — time(Time)
| model | run_timestamp | train PR-AUC | val PR-AUC | test PR-AUC | val precision@1000 | val fraud_found@1000 | test precision@1000 | test fraud_found@1000 | fraud rate (train/val/test) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgb_baseline | 2026-03-03 11:24:51 | 0.9983 | 0.8409 | **0.7660** | 0.052 | 52 | 0.044 | 44 | 0.00193 / 0.00131 / 0.00122 |

Artifacts:
- model: `results/models/xgb_20260303_112452.joblib`
- metrics: `results/metrics/xgb_20260303_112452.yaml`

---

### B) IEEE-CIS — time(TransactionDT)
| model | run_timestamp | scale_pos_weight | train PR-AUC | val PR-AUC | test PR-AUC | val precision@1000 | val fraud_found@1000 | test precision@1000 | test fraud_found@1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgb_baseline | 2026-03-03 12:04:30 | 27.4343 | 0.6465 | 0.4957 | **0.4852** | 0.839 | 839 | 0.840 | 840 |
| logreg_balanced | 2026-03-03 11:34:17 | – | 0.4327 | 0.3808 | 0.1635 | 0.681 | 681 | 0.019 | 19 |

Artifacts:
- xgb model: `results/models/xgb_20260303_120431.joblib` | metrics: `results/metrics/xgb_20260303_120431.yaml`
- logreg model: `results/models/logreg_20260303_113418.joblib` | metrics: `results/metrics/logreg_20260303_113418.yaml`

---

## Key findings (Phase 2)
- **Time-aware splits change the evaluation difficulty** and can shift class prevalence across splits (seen clearly on `creditcard.csv`).
- On **IEEE-CIS**, **Logistic Regression degrades sharply** on the later (test) period (`test_pr_auc ≈ 0.1635`, `fraud_found@1000 = 19`), while **XGBoost remains robust** (`test_pr_auc ≈ 0.4852`, `fraud_found@1000 = 840`).
- This supports the Phase 2 conclusion: **time/order-aware evaluation is necessary** to avoid overly optimistic results from random splitting and to reveal generalisation issues that look like drift.

---

## Reproducibility
- Runs log to: `results/metrics/runs.csv`
- Split indices saved under: `results/splits/`
- Model artifacts saved under: `results/models/`
- Per-run metrics saved under: `results/metrics/`