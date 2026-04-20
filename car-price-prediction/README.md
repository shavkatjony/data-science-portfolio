# Used Car Price Prediction — End-to-End ML Pipeline

Predicting second-hand car resale prices using a complete supervised learning pipeline. The project covers the full workflow from raw data through outlier removal, log transformation, multicollinearity analysis, feature engineering, and model evaluation — including 5-fold cross-validation to confirm generalization. Extended with business-level analysis of brand and body type pricing patterns.

---

## Dataset

- **File:** `data/used_cars.csv`
- **Records:** ~4,000+ used car listings
- **Features:** Brand, Body type, Mileage, Engine volume, Year, Registration status, Engine type
- **Target:** `Price` (continuous, log-transformed for modeling)

---

## Methods

- Quantile-based outlier detection and removal
- Log transformation of target variable to fix right-skew and linearize relationships
- VIF-based multicollinearity analysis (dropped `Year` due to VIF > 10)
- One-hot encoding with `drop_first=True` to avoid the dummy variable trap
- OLS regression via Statsmodels + Linear Regression via Scikit-learn
- 5-fold cross-validation for robustness testing
- Train/test split with R², RMSE, and MAE evaluation
- Residual analysis, predicted vs. actual scatter, error bucket distribution
- Brand and body type market analysis for business insights

---

## Results

| Metric | Train | Test |
|--------|-------|------|
| R² Score | ~0.84 | ~0.82 |
| RMSE (log scale) | ~0.19 | ~0.20 |
| MAE (log scale) | ~0.14 | ~0.15 |

Cross-validation R² was consistent across all 5 folds (~0.82–0.85), confirming stability.

---

## Key Findings

1. **Log transformation was essential** — raw price was heavily right-skewed; log-transform yielded near-normal distribution and valid OLS assumptions
2. **Year was dropped to fix multicollinearity** — it shared too much information with engine volume and mileage (VIF > 10)
3. **Mileage is the strongest price reducer** — largest negative coefficient, consistent with real-world depreciation
4. **BMW and Mercedes command premium resale** — brand dummies show significantly higher residual value for German premium brands
5. **No overfitting** — small train/test gap and stable cross-validation confirm generalization

---

## How to Run

```bash
cd used-car-price
pip install -r ../requirements.txt
jupyter notebook used_car_price.ipynb
```

---

## Notebook Structure

| # | Section | Description |
|---|---------|-------------|
| 1 | Setup & Imports | Libraries and configuration |
| 2 | Data Loading | CSV load, shape, dtypes, first look |
| 3 | Preprocessing | Drop irrelevant columns, handle missing values |
| 4 | Outlier Detection | Quantile-based filtering with rationale |
| 5 | Log Transformation | Fix skewness, linearize target |
| 6 | Multicollinearity | Correlation heatmap, VIF analysis |
| 7 | Dummy Variables | One-hot encoding with drop_first |
| 8 | OLS Regression | Statsmodels summary, coefficient analysis |
| 9 | Feature Importance | Standardized coefficients ranking |
| 10 | Predictions & Residuals | Predicted vs. actual, residual plots |
| 11 | Cross-Validation | 5-fold CV for robustness |
| 12 | Error Analysis | Error buckets, distribution |
| 13 | Business Insights | Brand and body type pricing analysis |
| 14 | Conclusion | Summary, findings, next steps |
