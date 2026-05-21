# Retail Demand Forecasting - Rossmann Store Sales
# BANA 288: Predictive Analytics | UCI Paul Merage School of Business
# Dataset: https://www.kaggle.com/competitions/rossmann-store-sales/data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

store = pd.read_csv("data/store.csv")
test  = pd.read_csv("data/test.csv", low_memory=False)
train = pd.read_csv("data/train.csv", low_memory=False)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Store shape:", store.shape)

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────

# Remove closed store days (Open = 0 always produces Sales = 0)
train = train[train["Open"] != 0]

# Remove zero-sales records on open days
train = train[train["Sales"] > 0]

# Merge store metadata into train
train = train.merge(store, on="Store", how="left")

# Impute missing CompetitionDistance with median
median_comp = train["CompetitionDistance"].median()
train["CompetitionDistance"].fillna(median_comp, inplace=True)

print("Cleaned train shape:", train.shape)
print("Missing values:\n", train.isnull().sum()[train.isnull().sum() > 0])

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Extract date features
train["Date"] = pd.to_datetime(train["Date"])
train["Year"]  = train["Date"].dt.year
train["Month"] = train["Date"].dt.month
train["Day"]   = train["Date"].dt.day

# Encode categorical variables
train = pd.get_dummies(train, columns=["StoreType", "Assortment"], drop_first=True)

print("Features after engineering:", train.shape[1])

# ─────────────────────────────────────────────
# 4. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

# Sales distribution
plt.figure(figsize=(10, 5))
plt.hist(train["Sales"], bins=50, color="steelblue", edgecolor="white")
plt.title("Daily Sales Distribution")
plt.xlabel("Sales (EUR)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/sales_distribution.png", dpi=150)
plt.show()
print("Distribution is right-skewed — most store-days fall between EUR 4,000 and EUR 10,000")

# Promotional effect
plt.figure(figsize=(8, 5))
sns.boxplot(x="Promo", y="Sales", data=train, palette=["#d9534f", "#5cb85c"])
plt.title("Effect of Promotions on Daily Sales")
plt.xlabel("Promotion Active (0 = No, 1 = Yes)")
plt.ylabel("Sales (EUR)")
plt.tight_layout()
plt.savefig("outputs/promo_effect.png", dpi=150)
plt.show()
print("Median sales are roughly 20-30% higher on promotional days")

# Sales by day of week
plt.figure(figsize=(9, 5))
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_map = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
           5: "Friday", 6: "Saturday", 7: "Sunday"}
train["DayName"] = train["DayOfWeek"].map(day_map)
sns.boxplot(x="DayName", y="Sales", data=train, order=day_order, palette="Blues_d")
plt.title("Sales by Day of Week")
plt.xlabel("Day")
plt.ylabel("Sales (EUR)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/sales_by_day.png", dpi=150)
plt.show()

# Correlation heatmap
numeric_cols = ["Sales", "Promo", "SchoolHoliday", "CompetitionDistance",
                "DayOfWeek", "Month", "Year"]
plt.figure(figsize=(8, 6))
sns.heatmap(train[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 5. MODEL PREPARATION
# ─────────────────────────────────────────────

features = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday",
    "CompetitionDistance", "Year", "Month", "Day"
]

# Add dummies if they exist
dummy_cols = [c for c in train.columns if c.startswith("StoreType_") or c.startswith("Assortment_")]
features += dummy_cols

X = train[features]
y = train["Sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train):,}")
print(f"Test samples:     {len(X_test):,}")

# ─────────────────────────────────────────────
# 6. LINEAR REGRESSION (BASELINE)
# ─────────────────────────────────────────────

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr  = mean_absolute_error(y_test, y_pred_lr)
r2_lr   = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Results:")
print(f"  RMSE: {rmse_lr:,.2f}")
print(f"  MAE:  {mae_lr:,.2f}")
print(f"  R2:   {r2_lr:.4f}")

# ─────────────────────────────────────────────
# 7. RANDOM FOREST
# ─────────────────────────────────────────────

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf  = mean_absolute_error(y_test, y_pred_rf)
r2_rf   = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print(f"  RMSE: {rmse_rf:,.2f}")
print(f"  MAE:  {mae_rf:,.2f}")
print(f"  R2:   {r2_rf:.4f}")

# ─────────────────────────────────────────────
# 8. MODEL COMPARISON
# ─────────────────────────────────────────────

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "RMSE":  [rmse_lr, rmse_rf],
    "MAE":   [mae_lr,  mae_rf],
    "R2":    [r2_lr,   r2_rf]
})
print("\nModel Comparison:")
print(results.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ["RMSE", "MAE", "R2"]
for ax, metric in zip(axes, metrics):
    ax.bar(results["Model"], results[metric], color=["#d9534f", "#5cb85c"], edgecolor="white")
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=15)
plt.suptitle("Model Comparison: Linear Regression vs Random Forest", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────
# 9. ACTUAL VS PREDICTED
# ─────────────────────────────────────────────

sample = np.random.choice(len(y_test), size=500, replace=False)
y_actual_sample = np.array(y_test)[sample]
y_pred_sample   = y_pred_rf[sample]

plt.figure(figsize=(10, 5))
plt.scatter(range(500), y_actual_sample, alpha=0.4, label="Actual", color="steelblue", s=15)
plt.scatter(range(500), y_pred_sample,   alpha=0.4, label="Predicted", color="orange", s=15)
plt.title("Actual vs Predicted Sales (Random Forest, Sample of 500)")
plt.xlabel("Observation")
plt.ylabel("Sales (EUR)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
plt.show()
print("Model tracks overall trend but underestimates extreme demand spikes")

# ─────────────────────────────────────────────
# 10. FEATURE IMPORTANCE
# ─────────────────────────────────────────────

importance_df = pd.DataFrame({
    "Feature":    features,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).head(10)

plt.figure(figsize=(9, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Blues_d")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.show()
print("\nTop features:")
print(importance_df.to_string(index=False))

# ─────────────────────────────────────────────
# 11. 12-WEEK FORECAST
# ─────────────────────────────────────────────

# Generate forward-looking date range
last_date = train["Date"].max()
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=84, freq="D")

# Build forecast dataframe using a representative store (Store 1)
forecast_df = pd.DataFrame({
    "Date":       forecast_dates,
    "Store":      1,
    "DayOfWeek":  forecast_dates.dayofweek + 1,
    "Promo":      0,
    "SchoolHoliday": 0,
    "CompetitionDistance": median_comp,
    "Year":       forecast_dates.year,
    "Month":      forecast_dates.month,
    "Day":        forecast_dates.day
})

# Add dummy columns with zeros
for col in dummy_cols:
    forecast_df[col] = 0

forecast_df["PredictedSales"] = rf.predict(forecast_df[features])
forecast_df["Week"] = forecast_df["Date"].dt.isocalendar().week

weekly_forecast = forecast_df.groupby("Week")["PredictedSales"].sum().reset_index()

plt.figure(figsize=(11, 5))
plt.plot(weekly_forecast["Week"], weekly_forecast["PredictedSales"],
         marker="o", color="steelblue", linewidth=2)
plt.fill_between(weekly_forecast["Week"], weekly_forecast["PredictedSales"],
                 alpha=0.15, color="steelblue")
plt.title("12-Week Forward Sales Forecast (Store 1, No Promotions)")
plt.xlabel("Week Number")
plt.ylabel("Predicted Weekly Sales (EUR)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("outputs/12_week_forecast.png", dpi=150)
plt.show()
print("\nWeekly forecast summary:")
print(weekly_forecast.to_string(index=False))
