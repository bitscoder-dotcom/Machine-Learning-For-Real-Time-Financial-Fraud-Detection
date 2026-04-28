# Fraud Dissertation Project

**Project title:** Machine Learning for Real-Time Financial Fraud Detection: Handling Imbalance, Drift, and Interpretability

This repository contains the implementation work for the fraud detection dissertation. The project trains and evaluates machine learning models for financial fraud detection, with focus on class imbalance, time-aware evaluation, drift, and interpretability.

---

## 1. Project Requirements

Install the required Python libraries before running the project:

```bash
pip install pandas numpy scikit-learn matplotlib joblib pyyaml shap
```

Main libraries used:

```text
pandas
numpy
scikit-learn
matplotlib
joblib
pyyaml
shap
```

---

## 2. Starting the Project

Open a terminal or command prompt and move into the project directory:

```bash
fraud-dissertation
```

If using a virtual environment, activate it first:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib joblib pyyaml shap
```

---

## 3. Running the Work

The project is organised around experiment scripts for training, evaluation, drift analysis, and explanation generation.

Typical workflow:

1. Prepare or confirm dataset paths.
2. Generate or load saved train/validation/test split files.
3. Train baseline models.
4. Evaluate model performance.
5. Generate plots, metrics, and explanations.
6. Review outputs inside the `results` folder.

Run the relevant Python scripts from the project root folder. For example:

```bash
python src/models/train_baseline.py
python src/models/train_xgb.py
```

If a script uses a configuration file, confirm that the dataset path and split path inside the config file are correct before running it.

---

## 4. Results Folder

All main outputs are stored inside the `results` folder.

Location:

```text
fraud-dissertation\results
```

The folder currently contains:

```text
results/
├── explanations/
├── metrics/
├── models/
├── plots/
└── splits/
```

### Folder Description

| Folder | Purpose |
|---|---|
| `results/explanations` | Stores SHAP outputs and interpretability files. |
| `results/metrics` | Stores model evaluation metrics, YAML files, CSV logs, and result summaries. |
| `results/models` | Stores trained model artefacts, usually saved with `joblib`. |
| `results/plots` | Stores generated charts and figures used in the dissertation and supporting document. |
| `results/splits` | Stores saved train/validation/test split indices for reproducibility. |

---

## 5. How to Locate Plots and Results

To view the generated plots, open:

```text
fraud-dissertation\results\plots
```

To view metric outputs, open:

```text
fraud-dissertation\results\metrics
```

To view trained models, open:

```text
fraud-dissertation\results\models
```

To view SHAP explanation outputs, open:

```text
fraud-dissertation\results\explanations
```

To view saved split files, open:

```text
fraud-dissertation\results\splits
```

---

## 6. Reproducibility Notes

The project uses saved split indices so that each model is evaluated on the same train, validation, and test rows. This helps ensure fair comparison across Logistic Regression, Random Forest, and XGBoost.

The pipeline also follows train-only preprocessing. Any scaling, resampling, or fitted transformation should be learned from the training data only, then applied to validation and test data.

This is important because the dissertation focuses on leakage-resistant evaluation.

---

## 7. Expected Outputs

After running the project scripts, expected outputs include:

- Trained model files in `results/models`
- Evaluation metric files in `results/metrics`
- Generated plots in `results/plots`
- SHAP explanation outputs in `results/explanations`
- Saved split index files in `results/splits`

These outputs support the main dissertation report and the supporting material document.

---

## 8. Project Focus

The implementation evaluates fraud detection models using operationally meaningful metrics, including:

- PR-AUC
- Recall at low false positive rate
- Precision@K
- FraudFound@K
- Inference latency
- SHAP explanation stability

The aim is not only to train accurate models, but to evaluate them under conditions closer to real fraud detection deployment.
