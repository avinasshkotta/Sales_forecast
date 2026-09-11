# Retail Sales Forecasting - Regression Project
# Dataset: stores_sales_forecasting.csv
# Encoding note: the supplied CSV is read with cp1252 because it contains
# characters that are not valid UTF-8.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. DATA LOADING
# ============================================================

FILE_PATH = "stores_sales_forecasting.csv"

try:
    df = pd.read_csv(FILE_PATH, encoding="cp1252")
except UnicodeDecodeError:
    df = pd.read_csv(FILE_PATH, encoding="latin1")

print("Dataset shape:", df.shape)
print(df.head())

# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe(include="all").T)

# ============================================================
# 3. DATA CLEANING
# ============================================================

df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.dropna(subset=["Order Date", "Sales"]).copy()
df = df.drop_duplicates().copy()

# Useful time features
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Quarter"] = df["Order Date"].dt.quarter
df["Month_Name"] = df["Order Date"].dt.strftime("%b")
df["Day_of_Week"] = df["Order Date"].dt.day_name()

print("\nShape after cleaning:", df.shape)

# ============================================================
# 4. DESCRIPTIVE ANALYSIS
# ============================================================

print("\nSales statistics:")
print(df["Sales"].describe())

print("\nTotal Sales:", round(df["Sales"].sum(), 2))
print("Total Profit:", round(df["Profit"].sum(), 2))
print("Total Quantity:", int(df["Quantity"].sum()))

print("\nSales by Category:")
print(df.groupby("Category")["Sales"].sum().sort_values(ascending=False))

print("\nSales by Region:")
print(df.groupby("Region")["Sales"].sum().sort_values(ascending=False))

print("\nTop 10 Products by Sales:")
print(
    df.groupby("Product Name")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

# ============================================================
# 5. DIAGNOSTIC ANALYSIS
# ============================================================

category_summary = (
    df.groupby("Category")
      .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
      .sort_values("Sales", ascending=False)
)

print("\nCategory Sales vs Profit:")
print(category_summary)

region_summary = (
    df.groupby("Region")
      .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
      .sort_values("Sales", ascending=False)
)

print("\nRegion Sales vs Profit:")
print(region_summary)

print("\nAverage Sales by Discount level:")
print(df.groupby("Discount")["Sales"].mean().sort_index())

# ============================================================
# 6. VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 5))
sns.histplot(df["Sales"], bins=40, kde=True)
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
monthly_sales_plot = df.set_index("Order Date").resample("MS")["Sales"].sum()
monthly_sales_plot.plot(marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 7. CORRELATION ANALYSIS
# ============================================================

corr_cols = ["Sales", "Quantity", "Discount", "Profit"]
corr = df[corr_cols].corr()

print("\nCorrelation matrix:")
print(corr)

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Scatter plot for the strongest practical sales driver in this dataset
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Quantity", y="Sales", alpha=0.6)
plt.title("Quantity vs Sales")
plt.xlabel("Quantity")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# ============================================================
# 8. TRANSACTION-LEVEL REGRESSION
# Predict Sales from Quantity, Discount, and selected categories.
# ============================================================

transaction_features = [
    col for col in ["Quantity", "Discount", "Category", "Sub-Category", "Region"]
    if col in df.columns
]

X = df[transaction_features]
y = df["Sales"]

categorical_features = [c for c in transaction_features if df[c].dtype == "object"]
numeric_features = [c for c in transaction_features if c not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

transaction_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

transaction_model.fit(X_train, y_train)
transaction_pred = transaction_model.predict(X_test)

transaction_mae = mean_absolute_error(y_test, transaction_pred)
transaction_rmse = np.sqrt(mean_squared_error(y_test, transaction_pred))
transaction_r2 = r2_score(y_test, transaction_pred)

print("\nTransaction-level Linear Regression")
print("MAE :", round(transaction_mae, 4))
print("RMSE:", round(transaction_rmse, 4))
print("R2  :", round(transaction_r2, 4))

plt.figure(figsize=(7, 6))
plt.scatter(y_test, transaction_pred, alpha=0.6)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales - Transaction Regression")
plt.tight_layout()
plt.show()

# ============================================================
# 9. MONTHLY SALES FORECASTING USING LAG REGRESSION
# ============================================================
# This is the main forecasting section.
# Sales are aggregated by month and lagged sales are used as predictors.

monthly = (
    df.set_index("Order Date")
      .resample("MS")["Sales"]
      .sum()
      .reset_index()
      .rename(columns={"Sales": "Monthly_Sales"})
)

monthly["Time_Index"] = np.arange(len(monthly))
monthly["Year"] = monthly["Order Date"].dt.year
monthly["Month"] = monthly["Order Date"].dt.month
monthly["Quarter"] = monthly["Order Date"].dt.quarter
monthly["Lag_1"] = monthly["Monthly_Sales"].shift(1)
monthly["Lag_2"] = monthly["Monthly_Sales"].shift(2)
monthly["Lag_3"] = monthly["Monthly_Sales"].shift(3)
monthly["Rolling_3"] = monthly["Monthly_Sales"].shift(1).rolling(3).mean()

forecast_features = [
    "Time_Index", "Year", "Month", "Quarter",
    "Lag_1", "Lag_2", "Lag_3", "Rolling_3"
]

forecast_data = monthly.dropna().copy()

# Time-order split: do not randomly shuffle time-series observations.
split_index = int(len(forecast_data) * 0.80)
train = forecast_data.iloc[:split_index].copy()
test = forecast_data.iloc[split_index:].copy()

forecast_model = LinearRegression()
forecast_model.fit(train[forecast_features], train["Monthly_Sales"])

test_pred = forecast_model.predict(test[forecast_features])

forecast_mae = mean_absolute_error(test["Monthly_Sales"], test_pred)
forecast_rmse = np.sqrt(mean_squared_error(test["Monthly_Sales"], test_pred))
forecast_r2 = r2_score(test["Monthly_Sales"], test_pred)

print("\nMonthly Sales Forecast Regression")
print("MAE :", round(forecast_mae, 4))
print("RMSE:", round(forecast_rmse, 4))
print("R2  :", round(forecast_r2, 4))

test_results = test[["Order Date", "Monthly_Sales"]].copy()
test_results["Predicted_Sales"] = test_pred

print("\nActual vs Predicted monthly sales:")
print(test_results.to_string(index=False))

plt.figure(figsize=(12, 5))
plt.plot(train["Order Date"], train["Monthly_Sales"], label="Train")
plt.plot(test["Order Date"], test["Monthly_Sales"], label="Actual Test", marker="o")
plt.plot(test["Order Date"], test_pred, label="Predicted Test", marker="x")
plt.title("Monthly Sales - Actual vs Predicted")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 10. FUTURE SALES FORECAST
# ============================================================
# Recursive 3-month forecast. Each predicted month becomes a lag
# for the following month.

future_steps = 3
history = monthly[["Order Date", "Monthly_Sales"]].copy()

future_rows = []

for _ in range(future_steps):
    next_date = history["Order Date"].max() + pd.offsets.MonthBegin(1)

    values = history["Monthly_Sales"].tolist()
    lag1 = values[-1]
    lag2 = values[-2]
    lag3 = values[-3]
    rolling3 = np.mean(values[-3:])

    row = pd.DataFrame([{
        "Order Date": next_date,
        "Time_Index": len(history),
        "Year": next_date.year,
        "Month": next_date.month,
        "Quarter": next_date.quarter,
        "Lag_1": lag1,
        "Lag_2": lag2,
        "Lag_3": lag3,
        "Rolling_3": rolling3,
    }])

    predicted_sales = forecast_model.predict(row[forecast_features])[0]
    predicted_sales = max(0, predicted_sales)

    row["Monthly_Sales"] = predicted_sales
    history = pd.concat(
        [history, row[["Order Date", "Monthly_Sales"]]],
        ignore_index=True
    )
    future_rows.append([next_date, predicted_sales])

future_forecast = pd.DataFrame(
    future_rows, columns=["Forecast_Month", "Forecast_Sales"]
)

print("\nNext 3 months sales forecast:")
print(future_forecast.to_string(index=False))

plt.figure(figsize=(12, 5))
plt.plot(monthly["Order Date"], monthly["Monthly_Sales"], label="Historical Sales")
plt.plot(
    future_forecast["Forecast_Month"],
    future_forecast["Forecast_Sales"],
    marker="o",
    linestyle="--",
    label="Forecast"
)
plt.title("Future Monthly Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 11. PRODUCT-LEVEL SALES SUMMARY
# ============================================================

product_sales = (
    df.groupby(["Product ID", "Product Name"])["Sales"]
      .sum()
      .sort_values(ascending=False)
      .reset_index()
)

print("\nTop 10 products:")
print(product_sales.head(10).to_string(index=False))

# ============================================================
# 12. BUSINESS INSIGHTS
# ============================================================

best_category = category_sales.idxmax()
best_region = df.groupby("Region")["Sales"].sum().idxmax()
best_product = product_sales.iloc[0]["Product Name"]

print("\nBUSINESS INSIGHTS")
print("-----------------")
print("Best-selling category:", best_category)
print("Highest-sales region :", best_region)
print("Top product          :", best_product)
print("Total sales          :", round(df["Sales"].sum(), 2))
print("Total profit         :", round(df["Profit"].sum(), 2))

print("\nInterpretation:")
print("- Historical monthly sales are used to estimate future sales.")
print("- Lagged sales capture recent demand patterns.")
print("- Forecasts should support inventory planning, not replace business judgment.")
print("- Because this dataset has no Store column, forecasting is performed at overall monthly level.")
