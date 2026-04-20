# GPA Prediction with Dummy Variables & OLS Regression

Predicting student GPA from SAT scores and class attendance using Ordinary Least Squares regression. This project demonstrates how categorical predictors (attended vs. not attended) can be encoded as dummy variables and how their inclusion creates parallel regression lines — revealing that attendance has a meaningful positive effect on GPA independent of test scores.

---

## Dataset

- **File:** `data/gpa_dummy.csv`
- **Records:** Student-level observations with SAT scores, attendance status, and GPA
- **Features:** `SAT` (continuous), `Attendance` (binary: Yes/No)
- **Target:** `GPA` (continuous, 0–4 scale)

---

## Methods

- Dummy variable encoding for categorical attendance feature
- OLS regression via Statsmodels (with and without dummy variables)
- Model diagnostics: VIF, Durbin-Watson, Breusch-Pagan, Q-Q plot
- Feature scaling (StandardScaler) to reduce condition number
- Train/test split (80/20) with RMSE evaluation
- Prediction for hypothetical students (Bob: SAT 1700, no attendance / Alice: SAT 1670, attended)

---

## Results

| Metric | Value |
|--------|-------|
| R² (adjusted) | Reported in notebook |
| RMSE (test set) | Reported in notebook |
| Condition number (scaled) | Significantly reduced vs. unscaled |
| VIF (all features) | Below problematic thresholds |

### Key Predictions

| Student | SAT | Attended | Predicted GPA |
|---------|-----|----------|---------------|
| Bob | 1700 | No | ~3.02 |
| Alice | 1670 | Yes | ~3.20 |

---

## Key Findings

1. **Attendance matters more than raw SAT** — Alice's lower SAT score is offset by the attendance effect, giving her a higher predicted GPA than Bob
2. **Dummy variables create a parallel shift** — two distinct regression lines with the same slope but different intercepts
3. **Scaling improves numerical stability** — standardizing SAT dramatically reduces the condition number without changing predictions
4. **Model passes all diagnostics** — Durbin-Watson, Breusch-Pagan, VIF, and Q-Q plot all confirm a well-specified model

---

## How to Run

```bash
cd gpa-prediction
pip install -r ../requirements.txt
jupyter notebook gpa_prediction.ipynb
```

---

## Notebook Structure

| # | Section | Description |
|---|---------|-------------|
| 1 | Setup & Imports | Libraries and display settings |
| 2 | Data Loading | Load CSV, inspect shape and dtypes |
| 3 | EDA | Descriptive statistics, distributions |
| 4 | OLS Baseline | Regression without dummy variable |
| 5 | Visualizations | Conditional regression lines by attendance |
| 6 | Model Diagnostics | VIF, Durbin-Watson, Breusch-Pagan, Q-Q |
| 7 | Train/Test Split | 80/20 split, RMSE evaluation |
| 8 | Scaled OLS | Standardized SAT for numerical stability |
| 9 | Predictions | GPA estimates for new students |
| 10 | Conclusion | Summary of findings and takeaways |
