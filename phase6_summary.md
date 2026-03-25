# Phase 6 Summary — Interpretability with SHAP

## Goal
Examine whether the best-performing fraud detection model can provide interpretable explanations, and test whether its main explanatory drivers remain stable across later time windows.

## Main model
- **XGBoost**
- Chosen because earlier phases showed it offered the best balance between predictive strength and inference speed.

## Interpretability method
- **SHAP (SHapley Additive exPlanations)** was used to explain the model.
- A **fixed model trained on T1** was used so that explanations on later windows could be compared directly without mixing model updates with explanation changes.

## Outputs produced
1. **Global feature importance plot**
2. **Local explanation for a correctly flagged fraud**
3. **Local explanation for a false positive**
4. **Sanity check**
   - Do explanations match intuition?
   - Are top features stable across time windows?

## Evaluation windows
The same rolling windows from Phase 5 were used:
- **T1** = training window
- **T2** = first later test window
- **T3** = second later test window

Global SHAP explanations were produced separately for **T2** and **T3**, and the feature rankings were compared.

---

## IEEE-CIS results

### Window performance with the fixed model
- **T2**
  - PR-AUC = **0.4168**
  - Recall@1%FPR = **0.3442**
- **T3**
  - PR-AUC = **0.4379**
  - Recall@1%FPR = **0.3789**

### SHAP stability summary
- **Top-10 overlap count:** 10
- **Jaccard similarity:** 1.0000
- **Spearman rank correlation:** 0.9973

### Shared top features across T2 and T3
- `TransactionAmt`
- `TransactionDT`
- `D1`
- `C1`
- `C2`
- `C4`
- `C5`
- `C12`
- `C13`
- `C14`

### IEEE-CIS interpretation
- The model’s global explanations were **extremely stable** across T2 and T3.
- The same top ten features appeared in both windows.
- This suggests that the model relied on a **consistent set of explanatory drivers** over adjacent future periods.
- Important features included transaction amount, time or order position, and engineered behavioural variables, which are plausible fraud-related signals.

### Sanity check for IEEE-CIS
- The explanations broadly match intuition.
- `TransactionAmt` is a plausible fraud driver because unusual transaction values often carry risk.
- `TransactionDT` suggests that temporal position itself contains predictive information, which is reasonable in a drifting environment.
- `D*` and `C*` variables likely capture behavioural or aggregate patterns, making their importance plausible in fraud detection.

---

## CreditCard results

### Window performance with the fixed model
- **T2**
  - PR-AUC = **0.8490**
  - Recall@1%FPR = **0.9286**
- **T3**
  - PR-AUC = **0.7507**
  - Recall@1%FPR = **0.8077**

### SHAP stability summary
- **Top-10 overlap count:** 9
- **Jaccard similarity:** 0.8182
- **Spearman rank correlation:** 0.9982

### Shared top features across T2 and T3
- `V1`
- `V4`
- `V10`
- `V11`
- `V12`
- `V14`
- `V16`
- `V17`
- `V19`

### CreditCard interpretation
- Explanations were also **highly stable** across time, though slightly less perfectly aligned than IEEE-CIS.
- The top-10 overlap was 9 out of 10, and the overall ranking correlation remained extremely high.
- This suggests that the model relied on a **largely stable set of latent patterns** over time.

### Sanity check for CreditCard
- The CreditCard dataset uses anonymised variables (`V1–V28`), so it is not possible to assign direct business meaning to individual features.
- Because of that, the sanity check focuses on **consistency rather than domain semantics**.
- The high overlap and rank correlation indicate that the model’s reasoning was not highly unstable across adjacent future windows.

---

## Local explanations
Local SHAP explanations were produced for:
- one **correctly flagged fraud**
- one **false positive**

These outputs show which features pushed the model toward a fraud decision and which features contributed to an incorrect alert. This provides case-level transparency and supports the claim that the model is not a complete black box.

### Interpretation of local explanations
- The **correct fraud case** shows a combination of features pushing the score toward the fraud class.
- The **false positive case** shows that the model can be influenced by suspicious-looking patterns that resemble fraud but belong to legitimate transactions.
- Together, these examples help explain both successful detection and model error.

---

## Key findings (Phase 6)
1. **The best model was explainable with SHAP at both global and local levels.**
2. **Global explanations were highly stable across time windows** on both datasets.
3. **IEEE-CIS showed especially strong stability**, with identical top ten features across T2 and T3.
4. **CreditCard also showed strong stability**, though one top feature changed between windows.
5. The explanations broadly matched intuition where domain meaning was available, especially on IEEE-CIS.
6. On CreditCard, feature anonymisation limited semantic interpretation, so stability itself was the main sanity check.

---

## Practical conclusion
Phase 6 shows that the best-performing XGBoost model can provide useful explanations for both overall behaviour and individual alerts. The model’s main explanatory drivers remained largely stable across later time windows, increasing confidence that its decisions were not based on random or highly unstable patterns.

---

## Reproducibility
Outputs saved include:
- `results/plots/phase6_ieee_t2_global_shap.png`
- `results/plots/phase6_ieee_t3_global_shap.png`
- `results/plots/phase6_ieee_correct_flagged_fraud_t3.png`
- `results/plots/phase6_ieee_false_positive_t3.png`
- `results/metrics/phase6_ieee_t2_global_shap.csv`
- `results/metrics/phase6_ieee_t3_global_shap.csv`
- `results/metrics/phase6_ieee_stability_summary.txt`
- `results/metrics/phase6_ieee_window_metrics.csv`

and

- `results/plots/phase6_creditcard_t2_global_shap.png`
- `results/plots/phase6_creditcard_t3_global_shap.png`
- `results/plots/phase6_creditcard_correct_flagged_fraud_t3.png`
- `results/plots/phase6_creditcard_false_positive_t3.png`
- `results/metrics/phase6_creditcard_t2_global_shap.csv`
- `results/metrics/phase6_creditcard_t3_global_shap.csv`
- `results/metrics/phase6_creditcard_stability_summary.txt`
- `results/metrics/phase6_creditcard_window_metrics.csv`