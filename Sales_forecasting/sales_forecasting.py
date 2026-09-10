# =============================================================================
# RETAIL SALES FORECASTING - REGRESSION ANALYSIS
# Dataset: mock_kaggle.csv
# Columns in source dataset:
#   data    -> Date
#   venda   -> Sales
#   estoque -> Stock
#   preco   -> Price
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# STEP 1. DATA LOADING
# -----------------------------------------------------------------------------
FILE_PATH = "mock_kaggle.csv"

df = pd.read_csv(FILE_PATH)

print("=" * 80)
print("STEP 1 - DATA LOADING")
print("=" * 80)
print("Shape:", df.shape)
print(df.head())

# -----------------------------------------------------------------------------
# STEP 2. DATA UNDERSTANDING
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 2 - DATA UNDERSTANDING")
print("=" * 80)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe())

# -----------------------------------------------------------------------------
# STEP 3. DATA CLEANING
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 3 - DATA CLEANING")
print("=" * 80)

df.columns = df.columns.str.strip()

# Rename source columns to English names for readability
column_mapping = {
    "data": "Date",
    "venda": "Sales",
    "estoque": "Stock",
    "preco": "Price"
}
df = df.rename(columns=column_mapping)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

for col in ["Sales", "Stock", "Price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove invalid rows and duplicates
df = df.dropna(subset=["Date", "Sales", "Stock", "Price"])
df = df.drop_duplicates()

# Sales, stock and price cannot be negative
df = df[(df["Sales"] >= 0) & (df["Stock"] >= 0) & (df["Price"] >= 0)]

df = df.sort_values("Date").reset_index(drop=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned shape:", df.shape)
print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())

# -----------------------------------------------------------------------------
# STEP 4. FEATURE ENGINEERING
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 4 - FEATURE ENGINEERING")
print("=" * 80)

# Calendar features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["DayOfMonth"] = df["Date"].dt.day
df["Quarter"] = df["Date"].dt.quarter

# Lag features use only previous observations.
df["Sales_Lag_1"] = df["Sales"].shift(1)
df["Sales_Lag_7"] = df["Sales"].shift(7)
df["Sales_Rolling_7"] = df["Sales"].shift(1).rolling(7).mean()

# Previous-day stock and price avoid using future information.
df["Stock_Lag_1"] = df["Stock"].shift(1)
df["Price_Lag_1"] = df["Price"].shift(1)

df_model = df.dropna().copy()

print("Modeling rows:", len(df_model))
print("\nFeature columns:")
print([
    "Sales_Lag_1", "Sales_Lag_7", "Sales_Rolling_7",
    "Stock_Lag_1", "Price_Lag_1",
    "Year", "Month", "DayOfWeek", "DayOfMonth", "Quarter"
])

# -----------------------------------------------------------------------------
# STEP 5. DESCRIPTIVE ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 5 - DESCRIPTIVE ANALYSIS")
print("=" * 80)

print("\nSales statistics:")
print(df["Sales"].describe())

print("\nStock statistics:")
print(df["Stock"].describe())

print("\nPrice statistics:")
print(df["Price"].describe())

print("\nAverage sales by month:")
print(df.groupby("Month")["Sales"].mean().round(2))

print("\nTotal sales by year:")
print(df.groupby("Year")["Sales"].sum())

print("\nTop 10 sales days:")
print(df.nlargest(10, "Sales")[["Date", "Sales", "Stock", "Price"]])

# -----------------------------------------------------------------------------
# STEP 6. DIAGNOSTIC ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 6 - DIAGNOSTIC ANALYSIS")
print("=" * 80)

analysis_cols = ["Sales", "Stock", "Price", "Sales_Lag_1", "Sales_Lag_7"]
correlation = df_model[analysis_cols].corr()

print("\nCorrelation matrix:")
print(correlation.round(3))

# -----------------------------------------------------------------------------
# STEP 7. TIME-SERIES VISUALIZATION
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 7 - DATA VISUALIZATION")
print("=" * 80)

plt.figure(figsize=(14, 6))
plt.plot(df["Date"], df["Sales"], label="Actual Sales")
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# STEP 8. CORRELATION HEATMAP
# -----------------------------------------------------------------------------
plt.figure(figsize=(9, 7))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix - Sales Drivers")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# STEP 9. PREPARE DATA FOR REGRESSION
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 9 - FEATURE PREPARATION")
print("=" * 80)

features = [
    "Sales_Lag_1",
    "Sales_Lag_7",
    "Sales_Rolling_7",
    "Stock_Lag_1",
    "Price_Lag_1",
    "Year",
    "Month",
    "DayOfWeek",
    "DayOfMonth",
    "Quarter"
]

X = df_model[features]
y = df_model["Sales"]

# Time-series split: do NOT randomly shuffle the data.
split_index = int(len(df_model) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))

# -----------------------------------------------------------------------------
# STEP 10. MACHINE LEARNING - LINEAR REGRESSION
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 10 - MACHINE LEARNING: LINEAR REGRESSION")
print("=" * 80)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nIntercept:", round(model.intercept_, 4))

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values("Coefficient", ascending=False)

print("\nRegression coefficients:")
print(coefficients.to_string(index=False))

# -----------------------------------------------------------------------------
# STEP 11. MODEL EVALUATION
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 11 - MODEL EVALUATION")
print("=" * 80)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# -----------------------------------------------------------------------------
# STEP 12. ACTUAL VS PREDICTED VISUALIZATION
# -----------------------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(df_model["Date"].iloc[split_index:], y_test.values,
         label="Actual Sales")
plt.plot(df_model["Date"].iloc[split_index:], y_pred,
         label="Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Regression relationship
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)

line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())
plt.plot([line_min, line_max], [line_min, line_max],
         linestyle="--", label="Perfect Prediction")

plt.title("Actual Sales vs Predicted Sales")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# STEP 13. RESIDUAL ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 13 - RESIDUAL ANALYSIS")
print("=" * 80)

residuals = y_test.values - y_pred

print("Mean residual:", round(residuals.mean(), 4))
print("Residual std :", round(residuals.std(), 4))

plt.figure(figsize=(12, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.title("Residuals vs Predicted Sales")
plt.xlabel("Predicted Sales")
plt.ylabel("Residual")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# STEP 14. FUTURE SALES FORECAST
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 14 - FUTURE SALES FORECAST")
print("=" * 80)

# Forecast horizon can be changed.
FORECAST_DAYS = 30

history = df[["Date", "Sales", "Stock", "Price"]].copy()

# For future dates, the model uses historical sales lags and calendar features.
# Since future stock/price are unknown, their most recent observed values are
# carried forward as a simple baseline assumption.
last_date = history["Date"].max()
last_stock = history["Stock"].iloc[-1]
last_price = history["Price"].iloc[-1]

future_rows = []

sales_history = history["Sales"].tolist()

for i in range(1, FORECAST_DAYS + 1):
    future_date = last_date + pd.Timedelta(days=i)

    lag_1 = sales_history[-1]
    lag_7 = sales_history[-7] if len(sales_history) >= 7 else sales_history[0]
    rolling_7 = np.mean(sales_history[-7:])

    row = {
        "Sales_Lag_1": lag_1,
        "Sales_Lag_7": lag_7,
        "Sales_Rolling_7": rolling_7,
        "Stock_Lag_1": last_stock,
        "Price_Lag_1": last_price,
        "Year": future_date.year,
        "Month": future_date.month,
        "DayOfWeek": future_date.dayofweek,
        "DayOfMonth": future_date.day,
        "Quarter": future_date.quarter
    }

    predicted_sales = max(0, float(model.predict(pd.DataFrame([row])[features])[0]))

    future_rows.append({
        "Date": future_date,
        "Forecast_Sales": predicted_sales
    })

    # Recursive forecasting: next prediction becomes the next lag value.
    sales_history.append(predicted_sales)

forecast_df = pd.DataFrame(future_rows)

print("\n30-day sales forecast:")
print(forecast_df.to_string(index=False))

print("\nTotal forecast sales:", round(forecast_df["Forecast_Sales"].sum(), 2))
print("Average daily forecast:", round(forecast_df["Forecast_Sales"].mean(), 2))

# -----------------------------------------------------------------------------
# STEP 15. FORECAST VISUALIZATION
# -----------------------------------------------------------------------------
plt.figure(figsize=(14, 6))

recent_history = df.tail(90)

plt.plot(recent_history["Date"], recent_history["Sales"],
         label="Historical Sales")
plt.plot(forecast_df["Date"], forecast_df["Forecast_Sales"],
         linestyle="--", label="30-Day Forecast")

plt.title("Sales Forecast - Historical vs Future")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# STEP 16. BUSINESS CONCLUSION
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 16 - BUSINESS CONCLUSION")
print("=" * 80)

print(f"""
1. The dataset contains {len(df):,} cleaned daily observations.
2. Average daily sales: {df["Sales"].mean():.2f}.
3. Maximum observed daily sales: {df["Sales"].max():.2f}.
4. Linear Regression R² on the chronological test set: {r2:.4f}.
5. Average forecast sales for the next {FORECAST_DAYS} days:
   {forecast_df["Forecast_Sales"].mean():.2f} units/day.

Business use:
- Forecasts can support inventory replenishment decisions.
- Higher expected sales can trigger earlier restocking.
- Lower expected sales can reduce over-stocking risk.
- Forecast accuracy should be monitored as new transaction data becomes available.

Important assumption:
Future stock and price are unknown in advance, so this demonstration carries
the latest observed stock and price forward. In production, planned prices,
promotions, holidays, store information, and inventory constraints should be
included when available.
""")
