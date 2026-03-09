# Phase 3–4 Summary — Strong Baselines + Imbalance Methods (Time-Aware Evaluation)

## Goal
Establish credible baseline performance under a deployment-like protocol and then test whether imbalance treatments improve results under strict false-alarm constraints.

**Evaluation protocol (both phases)**
- **Time-aware split**: train on earlier data, validate/test on later data
- **Operating constraint**: pick a single **threshold on validation** such that **FPR ≤ 1%**, then apply unchanged to test
- **Metrics**:
  - PR-AUC (ranking quality under imbalance)
  - Recall@FPR=1% (low-false-alarm operating regime)
  - Precision@1000 and FraudFound@1000 (fixed analyst review budget)
  - CPU inference latency (ms/transaction)

---

## Phase 3 — Strong baselines

### CreditCard (time(Time), target FPR=1%, K=1000)

| Model | val PR-AUC | test PR-AUC | val Recall@1%FPR | test Recall@1%FPR | val Precision@1000 | test Precision@1000 | val FraudFound@1000 | test FraudFound@1000 | test ms/txn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8686 | **0.7744** | 0.9286 | **0.8462** | **0.053** | **0.045** | **53** | **45** | 0.00953 |
| XGBoost | 0.8409 | 0.7660 | 0.9286 | 0.8077 | 0.052 | 0.044 | 52 | 44 | 0.00417 |
| Logistic Regression | 0.8394 | 0.7069 | 0.9107 | 0.8077 | 0.052 | 0.044 | 52 | 44 | **0.00052** |

**Phase 3 finding (CreditCard).**
Random Forest is strongest on the time split, but Logistic Regression is orders of magnitude faster. XGBoost provides a middle ground between quality and latency.

---

### IEEE-CIS (time(TransactionDT), target FPR=1%, K=1000)

| Model | val PR-AUC | test PR-AUC | val Recall@1%FPR | test Recall@1%FPR | val Precision@1000 | test Precision@1000 | val FraudFound@1000 | test FraudFound@1000 | test ms/txn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | **0.5169** | **0.4901** | **0.4310** | **0.4207** | **0.855** | **0.848** | **855** | **848** | 0.07607 |
| XGBoost | 0.4957 | 0.4852 | 0.4145 | 0.3922 | 0.839 | 0.840 | 839 | 840 | **0.00725** |
| Logistic Regression | 0.3808 | 0.1635 | 0.3133 | **0.0000** | 0.681 | 0.019 | 681 | 19 | 0.00533 |

**Phase 3 finding (IEEE-CIS).**
Under the 1% FPR constraint on a later time segment, Logistic Regression fails to recover fraud cases (test Recall@1%FPR = 0). Random Forest and XGBoost are both robust, with Random Forest slightly stronger but ~10× slower than XGBoost at inference.

---

## Phase 4 — Imbalance methods (Logistic Regression only)

**Strategies compared**
- `logreg_cost`: class_weight="balanced" (cost-sensitive learning)
- `logreg_undersample`: random undersampling (train-only)
- `logreg_smote`: SMOTE oversampling (train-only)

### IEEE-CIS (time(TransactionDT), target FPR=1%, K=1000)

| Strategy | val PR-AUC | test PR-AUC | val Recall@1%FPR | test Recall@1%FPR | val Precision@1000 | test Precision@1000 | val FraudFound@1000 | test FraudFound@1000 | test ms/txn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg_cost | 0.3808 | 0.1635 | 0.3133 | 0.0000 | 0.681 | 0.019 | 681 | 19 | 0.00465 |
| logreg_undersample | 0.3851 | **0.1667** | 0.3162 | 0.0000 | **0.701** | 0.017 | **701** | 17 | **0.00428** |
| logreg_smote | **0.3941** | 0.1661 | **0.3258** | 0.0000 | 0.700 | 0.012 | 700 | 12 | 0.00533 |

**Phase 4 finding (IEEE-CIS).**
Resampling improves validation PR-AUC slightly, but **does not fix** the core issue: all Logistic Regression variants still achieve **test Recall@1%FPR = 0.0** under time-ordered evaluation. This indicates that imbalance handling alone is insufficient for robust low-FPR detection on IEEE-CIS, and motivates stronger nonlinear baselines for subsequent drift and interpretability experiments.

---

### CreditCard (time(Time), target FPR=1%, K=1000)

| Strategy | val PR-AUC | test PR-AUC | val Recall@1%FPR | test Recall@1%FPR | val Precision@1000 | test Precision@1000 | val FraudFound@1000 | test FraudFound@1000 | test ms/txn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg_undersample | 0.7648 | 0.6273 | 0.9107 | **0.8269** | 0.052 | **0.045** | 52 | **45** | **0.00047** |
| logreg_cost | **0.8394** | **0.7069** | **0.9107** | 0.8077 | 0.052 | 0.044 | 52 | 44 | 0.00058 |
| logreg_smote | 0.8201 | 0.6583 | 0.8929 | 0.7885 | 0.052 | 0.044 | 52 | 44 | 0.00051 |

**Phase 4 finding (CreditCard).**
Undersampling slightly improves low-FPR recall and top-K yield (45 vs 44 frauds found in the top 1000) but reduces overall ranking quality (lower PR-AUC). SMOTE does not provide a clear benefit over cost-sensitive learning on this dataset.

---

## Consolidated takeaways (Phases 3–4)
1) **Time-aware evaluation matters.** Under a strict low-FPR operating constraint, model rankings and practical usefulness change significantly.
2) **IEEE-CIS is more demanding.** Logistic Regression performs poorly under low-FPR constraints on the later period; nonlinear models (Random Forest / XGBoost) remain robust.
3) **Imbalance methods are dataset-dependent.** On CreditCard, undersampling gives a small low-FPR gain; on IEEE-CIS it does not resolve the low-FPR failure for Logistic Regression.
4) **Latency trade-offs are real.** Random Forest can be slightly stronger than XGBoost but is substantially slower at inference, especially on IEEE-CIS.

## Practical decision for next phases
- Use **XGBoost** as the main model for drift and interpretability work (strong performance with much lower latency than Random Forest).
- Keep **Random Forest** as a reference point for accuracy-latency trade-offs.
- Keep **Logistic Regression** as a lightweight baseline and as evidence of where linear models fail under drift + low-FPR operation.