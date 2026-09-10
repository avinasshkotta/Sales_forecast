# Retail Sales Forecasting

## Project Overview

**Project:** Sales Forecasting  
**Domain:** Retail  
**Analytics Type:** Descriptive, Diagnostic and Predictive Analytics  
**Machine Learning:** Linear Regression

This project forecasts future retail sales using historical transaction data.

The objective is to help businesses make better inventory and sales decisions. Over-stocking ties up capital, while under-stocking can result in lost sales.

---

## Business Problem

Retail businesses need to estimate future demand accurately.

If inventory is too high:

- Capital is locked in unsold stock
- Storage costs increase
- Products may become obsolete

If inventory is too low:

- Stock-outs can occur
- Sales opportunities are lost
- Customer satisfaction may decrease

Sales forecasting provides an estimate of future demand that can support inventory planning.

---

## Dataset

The project uses `mock_kaggle.csv`.

The dataset contains 937 daily observations covering **January 1, 2014 through July 31, 2016**.

### Dataset Columns

| Column | Meaning |
|---|---|
| `data` | Date |
| `venda` | Sales |
| `estoque` | Stock |
| `preco` | Price |

The Python program renames these fields to:

| Original | Program Name |
|---|---|
| `data` | `Date` |
| `venda` | `Sales` |
| `estoque` | `Stock` |
| `preco` | `Price` |

---

## Project Workflow

```text
1. Data Loading
        ↓
2. Data Understanding
        ↓
3. Data Cleaning
        ↓
4. Feature Engineering
        ↓
5. Descriptive Analysis
        ↓
6. Diagnostic Analysis
        ↓
7. Data Visualization
        ↓
8. Correlation Analysis
        ↓
9. Feature Preparation
        ↓
10. Linear Regression
        ↓
11. Model Evaluation
        ↓
12. Actual vs Predicted Visualization
        ↓
13. Residual Analysis
        ↓
14. Future Sales Forecast
        ↓
15. Forecast Visualization
        ↓
16. Business Conclusion
```

---

## Analytics Performed

### 1. Descriptive Analytics

Answers:

> What happened?

The project calculates:

- Mean
- Median-related distribution statistics
- Standard deviation
- Minimum and maximum
- Quartiles
- Monthly average sales
- Yearly total sales
- Highest-sales days

---

### 2. Diagnostic Analytics

Answers:

> What relationships or patterns can help explain sales?

Correlation analysis is performed between sales and important historical variables.

A correlation heatmap is generated for visual interpretation.

---

### 3. Predictive Analytics

Answers:

> What could happen in the future?

A Linear Regression model predicts sales using historical sales, stock, price and calendar features.

---

## Feature Engineering

The following features are created:

### Sales Features

- `Sales_Lag_1`
- `Sales_Lag_7`
- `Sales_Rolling_7`

### Inventory and Price Features

- `Stock_Lag_1`
- `Price_Lag_1`

### Calendar Features

- `Year`
- `Month`
- `DayOfWeek`
- `DayOfMonth`
- `Quarter`

Lagged variables are used to reduce future-information leakage.

---

## Machine Learning Model

### Linear Regression

The model estimates sales using a linear combination of the input features.

General form:

\[
y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n
\]

Where:

- `y` = predicted sales
- `b0` = intercept
- `b1...bn` = coefficients
- `x1...xn` = model features

---

## Train/Test Strategy

Because the data is time-dependent, the project does **not** randomly shuffle observations.

Instead:

- First 80% → Training
- Last 20% → Testing

This is more appropriate for a basic time-series forecasting workflow because future observations should not be used to train a model that predicts the past.

---

## Model Evaluation

The following metrics are used:

### MAE

Mean Absolute Error.

Lower values indicate smaller average prediction errors.

### RMSE

Root Mean Squared Error.

RMSE penalizes larger errors more heavily than MAE.

### R² Score

Measures the proportion of target variation explained by the regression model.

---

## Visualizations

The project generates:

1. Daily Sales Trend
2. Correlation Heatmap
3. Actual vs Predicted Sales
4. Actual Sales vs Predicted Sales Scatter Plot
5. Residuals vs Predicted Sales
6. Historical Sales vs 30-Day Forecast

---

## Future Forecast

The project generates a **30-day sales forecast**.

The forecast is recursive, meaning previous predictions can become lagged inputs for later forecast days.

Because future stock and price values are not available in the dataset, the demonstration carries the most recently observed stock and price forward.

For a production solution, future known information such as promotions, planned prices, holidays, inventory levels, stores and products should be incorporated.

---

## Installation

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Project Structure

```text
Retail-Sales-Forecasting/
│
├── mock_kaggle.csv
├── sales_forecasting.py
├── sales_forecasting_steps.md
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Open the project

```bash
cd Retail-Sales-Forecasting
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Run the Python program

```bash
python sales_forecasting.py
```

The program prints the analysis results and displays the visualizations.

---

## Business Benefits

The forecasting solution can help retailers:

- Estimate upcoming demand
- Plan inventory replenishment
- Reduce over-stocking
- Reduce stock-out risk
- Improve inventory turnover
- Support purchasing decisions
- Improve sales planning

---

## Limitations

This project is a demonstration of a regression-based forecasting workflow.

Potential improvements include:

- Store-level forecasting
- Product-level forecasting
- Holiday features
- Promotion/discount features
- Weather data
- Customer information
- Advanced time-series models
- Random Forest / Gradient Boosting
- XGBoost
- ARIMA/SARIMA
- Prophet
- Cross-validation designed for time series
- Hyperparameter tuning

The current dataset does not provide explicit store or product identifiers, so the implementation forecasts the overall daily sales series.

---

## Key Takeaway

The project demonstrates a complete retail analytics workflow:

**Historical Data → Cleaning → Descriptive Analysis → Diagnostic Analysis → Feature Engineering → Regression → Evaluation → Future Sales Forecast → Business Decision Support**

The final forecast can be used as an input to inventory planning, but business decisions should also consider operational constraints and additional future information.
