# Phase 5 Summary — Drift Testing with Rolling Evaluation

## Goal
Evaluate whether fraud detection performance remains stable over time under a deployment-like rolling protocol, and test a simple mitigation for temporal drift.

## Main model
- **XGBoost**
- Chosen as the main Phase 5 model because it provided strong performance in earlier phases while remaining much faster than Random Forest on IEEE-CIS.

## Evaluation design
A **rolling-origin evaluation** protocol was used to test temporal robustness.

### Time windows
Each dataset was divided chronologically into three windows:
- **T1** = earliest 70%
- **T2** = next 15%
- **T3** = final 15%

### Fixed model condition
- Train on **T1**
- Select threshold on a time-ordered validation slice inside **T1**
- Test on **T2**
- Test again on **T3** with the **same model and same threshold**

This condition measures how performance changes when the model is left unchanged.

### Mitigation condition: periodic retraining
- Train on **T1**, test on **T2**
- Retrain on **T1 + T2**, then test on **T3**

This provides a simple mitigation strategy for temporal change.

## Metrics used
- **PR-AUC**
- **Recall at the validation-selected 1% FPR threshold**

These were chosen to remain consistent with earlier phases and to reflect low-false-alarm operational performance.

---

## IEEE-CIS results

| Strategy | Train Window | Test Window | PR-AUC | Recall@1%FPR | Actual FPR |
|---|---|---:|---:|---:|---:|
| Fixed | T1 | T2 | 0.4168 | 0.3442 | 0.0084 |
| Fixed | T1 | T3 | 0.4379 | 0.3789 | 0.0100 |
| Periodic retraining | T1 | T2 | 0.4168 | 0.3442 | 0.0084 |
| Periodic retraining | T1+T2 | T3 | **0.4451** | **0.3938** | 0.0111 |

### IEEE-CIS interpretation
- Performance varied across later time windows, showing temporal instability.
- The fixed model did not collapse over time, but its performance changed between T2 and T3.
- **Periodic retraining improved T3 performance** relative to the fixed model:
  - PR-AUC improved from **0.4379** to **0.4451**
  - Recall improved from **0.3789** to **0.3938**
- This suggests that a simple retraining policy can recover some performance on later periods.

---

## CreditCard results

| Strategy | Train Window | Test Window | PR-AUC | Recall@1%FPR | Actual FPR |
|---|---|---:|---:|---:|---:|
| Fixed | T1 | T2 | 0.8490 | 0.9286 | 0.0095 |
| Fixed | T1 | T3 | 0.7507 | 0.8077 | 0.0080 |
| Periodic retraining | T1 | T2 | 0.8490 | 0.9286 | 0.0095 |
| Periodic retraining | T1+T2 | T3 | **0.7593** | 0.7885 | 0.0072 |

### CreditCard interpretation
- CreditCard showed a clearer drop from T2 to T3 under the fixed model:
  - PR-AUC fell from **0.8490** to **0.7507**
  - Recall fell from **0.9286** to **0.8077**
- This indicates a more obvious temporal degradation than in IEEE-CIS.
- Periodic retraining gave a **small PR-AUC improvement** on T3:
  - 0.7593 vs 0.7507
- However, retraining did **not** improve Recall@1%FPR on T3:
  - 0.7885 vs 0.8077

---

## Key findings (Phase 5)
1. **Rolling evaluation revealed time-dependent performance variation** on both datasets.
2. **CreditCard showed clearer temporal degradation** under a fixed model.
3. **IEEE-CIS showed more mixed temporal behaviour**, but periodic retraining still improved the final-window results.
4. **Periodic retraining helped, but not uniformly across datasets or metrics.**
   - On IEEE-CIS, it improved both PR-AUC and Recall on T3.
   - On CreditCard, it improved PR-AUC slightly but did not improve low-FPR Recall.
5. These results support the idea that **fraud models should be monitored over time rather than treated as static systems**.

---

## Practical conclusion
Phase 5 shows that fraud detection performance is not fully stable over time under realistic chronological evaluation. A simple mitigation based on **periodic retraining** can provide useful gains, but the size and consistency of the benefit depend on the dataset and the performance metric considered.

---

## Reproducibility
Outputs saved:
- `results/metrics/phase5_ieee_rolling_results.csv`
- `results/metrics/phase5_creditcard_rolling_results.csv`
- `results/plots/phase5_ieee_pr_auc_over_time.png`
- `results/plots/phase5_ieee_recall_at_1pct_fpr_over_time.png`
- `results/plots/phase5_creditcard_pr_auc_over_time.png`
- `results/plots/phase5_creditcard_recall_at_1pct_fpr_over_time.png`