# Analysis Appendix

Full technical documentation for the Retail Demand Forecasting project. This document covers the analytical decisions, model justifications, and findings in detail.

---

## 1. Project Background

Rossmann is one of Europe's largest drugstore chains, operating over 3,000 stores across seven countries. At that scale, even modest inefficiencies in inventory allocation or staffing translate to meaningful revenue loss.

Without reliable short-term forecasts, store managers default to rules of thumb and historical averages, which fail systematically during promotional periods when demand spikes are both larger and more variable. The downstream costs are real: overstocking ties up working capital, while stockouts during high-demand periods directly reduce revenue and erode customer trust.

---

## 2. Data Cleaning Decisions

**Closed store days removed:** Observations where Open = 0 were excluded. Closed-day sales are always zero and would distort model learning if included since the model would attempt to explain zero sales through feature patterns that don't apply on operational days.

**Missing CompetitionDistance:** Imputed with the median across all stores. This is a conservative assumption that avoids introducing extreme values. A more rigorous approach would examine whether missingness is random or correlated with store characteristics before choosing an imputation strategy.

**Zero-sales on open days:** Any record where Sales = 0 for an open store was investigated and removed to avoid artificially suppressing model estimates for operational days.

---

## 3. Feature Engineering Decisions

**Date decomposition:** Year, Month, and DayOfWeek were extracted from the raw Date field. This converts an otherwise unusable date string into actionable predictors that the model can learn from. Feature extraction from dates consistently yields returns in model performance at modest engineering cost.

**DayOfWeek retained as ordinal:** Rather than one-hot encoding DayOfWeek, it was retained as an integer (1-7). Given the clear ordinal structure of weekdays, this representation is reasonable and reduces dimensionality.

**Promotion interaction not formalized:** An implicit interaction between Promo and DayOfWeek exists in the dataset (promotions on weekends vs. weekdays behave differently) but was not engineered as an explicit interaction term. This is a path to potential improvement in a future iteration.

---

## 4. Model Selection Rationale

**Linear Regression as baseline:** Required to establish a performance floor. Without it, there is no meaningful way to evaluate whether 0.296 R-squared is good, acceptable, or disappointing. For this dataset, it is acceptable for planning purposes but indicates meaningful unexplained variance remains.

**Random Forest as primary model:**
- Makes no distributional assumptions, which handles the right-skewed sales distribution naturally
- Captures non-linear relationships and implicit interactions between variables
- Aggregates across many decision trees, reducing overfitting relative to a single tree
- Default hyperparameters used; tuning (number of trees, max depth, min leaf size) was not performed and represents a clear path to improved performance

---

## 5. Train-Test Split Limitation

The 80/20 random split used in this project is the most significant methodological limitation. Time-series data has temporal dependencies — past sales influence future sales — and a random split allows the model to learn from future data during training. A time-ordered split (earlier data for training, later data for testing) would produce more conservative and more realistic out-of-sample performance estimates.

If this model were deployed operationally, its real-world performance would likely be lower than the test-set metrics suggest.

---

## 6. Model Performance Interpretation

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Linear Regression | 2,863.07 | 2,074.46 | 0.1500 |
| Random Forest | 2,604.00 | 1,891.73 | 0.2962 |

**RMSE interpretation:** An average prediction error of EUR 2,604 per store per day. For a store generating EUR 7,000 in daily sales on average, this represents roughly 37% relative error. Sufficient for directional planning but not precise enough for tight inventory optimization.

**R-squared interpretation:** Random Forest explains approximately 29.6% of variance in daily sales. The remaining 70.4% reflects factors not captured in the current feature set — unobserved local events, weather, store-specific dynamics, and the inherent unpredictability of consumer behavior.

**Spike underestimation:** The model consistently underestimates sales during extreme high-demand days. This is expected behavior from ensemble models due to averaging and the limited representation of rare extreme events in the training data. For operational use, managers should apply buffer inventory during promotional periods rather than treating model predictions as exact forecasts.

---

## 7. Hypothesis Results

| Hypothesis | Result | Evidence |
|---|---|---|
| H1: Promotions significantly increase daily store sales | Supported | Median sales 20-30% higher on promo days; promotions are top feature by importance |
| H2: Sales follow strong temporal patterns | Supported | DayOfWeek and Month are significant predictors; clear weekly and seasonal cycles |
| H3: Competition proximity influences store performance | Supported | CompetitionDistance is second-ranked predictor; farther stores perform better |

---

## 8. Forecasting Methodology

The 12-week forward forecast uses the trained Random Forest model to generate daily predictions which are then aggregated into weekly totals. Weekly aggregation reduces noise from day-to-day fluctuations and provides a more stable planning horizon for inventory management, staffing decisions, and promotion scheduling.

Weekly forecasts are more actionable for operations teams than daily point estimates.

---

## 9. References

- Rossmann Store Sales. Kaggle Competition Dataset. https://www.kaggle.com/competitions/rossmann-store-sales
- Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32.
- Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.
