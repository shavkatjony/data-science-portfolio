# Bank Marketing Campaign — Logistic Regression Classifier

Predicting whether a bank customer will subscribe to a term deposit after a marketing call. The pipeline progresses from a single-predictor baseline to a multivariate model, with full evaluation using confusion matrices, ROC curves, AUC scores, and a separate held-out test set to verify generalization. Extended with odds ratio analysis, probability distributions, and actionable business insights for campaign optimization.

---

## Dataset

- **Files:** `data/bank_data_train.csv`, `data/bank_data_test.csv`
- **Records:** 500+ bank customers (separate train and test files)
- **Features:** Interest rate, credit status, previous campaign outcome, call duration, month indicators
- **Target:** `y` — whether the customer subscribed to a term deposit (binary: yes/no → 1/0)

---

## Methods

- Binary encoding of target variable (yes/no → 1/0)
- Simple logistic regression (duration only) as baseline
- Multivariate logistic regression with all relevant predictors
- Confusion matrix analysis (TP, TN, FP, FN breakdown)
- ROC curve and AUC score computation
- Separate test set evaluation for generalization check
- Odds ratio calculation and forest plot visualization
- Predicted probability distribution by class
- Business insight extraction for campaign strategy

---

## Results

| Metric | Baseline (duration only) | Multivariate | Test Set |
|--------|--------------------------|--------------|----------|
| Accuracy | ~80% | ~86% | ~86% |
| AUC | ~0.79 | ~0.93 | — |
| Pseudo R² | ~0.16 | ~0.34 | — |

Train and test accuracy are nearly identical (~86.29% vs ~86.04%) — no overfitting.

---

## Key Findings

1. **Duration is the strongest predictor** — longer calls strongly correlate with subscription, reflecting genuine customer interest
2. **Previous campaign success matters** — past subscribers are much more likely to subscribe again; the bank should prioritize re-engagement
3. **High interest rates reduce subscriptions** — negative coefficient on interest_rate suggests customers avoid locking in when market alternatives seem better
4. **March calls underperform** — significant negative effect; the bank should consider shifting campaign timing
5. **Model generalizes well** — near-identical train/test accuracy confirms reliability on unseen customers

---

## How to Run

```bash
cd bank-marketing
pip install -r ../requirements.txt
jupyter notebook bank_marketing.ipynb
```

---

## Notebook Structure

| # | Section | Description |
|---|---------|-------------|
| 1 | Setup & Imports | Libraries and configuration |
| 2 | Data Loading | CSV load, variable descriptions table |
| 3 | Preprocessing | Encode target variable, clean data |
| 4 | EDA | Exploratory statistics and distributions |
| 5 | Simple Logistic Regression | Baseline model with duration only |
| 6 | Multivariate Logistic Regression | Full model with all features |
| 7 | Confusion Matrix | Classification accuracy breakdown |
| 8 | ROC Curve & AUC | Discrimination ability visualization |
| 9 | Odds Ratios | Coefficient interpretation as odds |
| 10 | Probability Distributions | Predicted probabilities by class |
| 11 | Test Set Evaluation | Generalization on held-out data |
| 12 | Feature Analysis | Coefficient significance and direction |
| 13 | Business Insights | Actionable campaign recommendations |
| 14 | Conclusion | Summary, findings, next steps |
