# Retail Sales Forecasting Using Regression

## Business Problem

> **Over-stocking wastes capital; under-stocking loses sales — both are forecasting failures.**

Retail businesses need reliable estimates of future demand to balance inventory availability and inventory cost.

This project analyzes historical retail transaction data and uses regression-based predictive analysis to forecast future sales.

---

## Project Objective

The project aims to:

- Understand historical retail sales.
- Perform descriptive analysis.
- Perform diagnostic analysis.
- Analyze correlations.
- Create meaningful visualizations.
- Build a transaction-level Linear Regression model.
- Build a monthly sales forecasting regression model.
- Evaluate regression performance.
- Forecast the next three months.
- Generate business and inventory-planning insights.

---

## Dataset

The supplied dataset is a retail transaction dataset containing fields such as:

- Order ID
- Order Date
- Ship Date
- Ship Mode
- Customer ID
- Customer Name
- Segment
- Country
- City
- State
- Postal Code
- Region
- Product ID
- Category
- Sub-Category
- Product Name
- Sales
- Quantity
- Discount
- Profit

The supplied file contains **2,121 transaction records and 21 columns**.

### Important Dataset Note

The dataset contains Product and Region information but **does not contain a Store column**.

Therefore, the forecasting section forecasts **overall monthly sales**, while product-level sales are analyzed separately.

---

## Project Structure

```text
Retail_Sales_Forecasting/
│
├── stores_sales_forecasting.csv
├── retail_sales_forecasting.py
├── markdown_steps.md
├── README.md
└── requirements.txt
```

---

## Analysis Stages

### 1. Data Loading
Load the CSV file using pandas.

### 2. Data Understanding
Inspect dimensions, columns, data types, statistics, and dataset information.

### 3. Data Cleaning
- Convert dates.
- Convert numerical columns.
- Check missing values.
- Remove duplicates.
- Remove records with invalid required dates/sales.
- Create time-based features.

### 4. Descriptive Analysis
Analyze:

- Total Sales
- Total Profit
- Total Quantity
- Sales distribution
- Category sales
- Regional sales
- Top-selling products

### 5. Diagnostic Analysis
Investigate why sales and profit vary using:

- Category
- Region
- Discount

### 6. Data Visualization
The project includes:

- Sales distribution histogram
- Category sales bar chart
- Monthly sales line chart
- Correlation heatmap
- Quantity vs Sales scatter plot
- Actual vs predicted regression plot
- Monthly forecast visualization

### 7. Correlation Analysis
Analyze relationships between:

- Sales
- Quantity
- Discount
- Profit

### 8. Transaction-Level Regression

A Linear Regression model predicts transaction-level Sales using:

- Quantity
- Discount
- Category
- Sub-Category
- Region

Categorical variables are converted using One-Hot Encoding.

### 9. Monthly Sales Forecasting

Sales are aggregated by month.

Forecasting features include:

- Time Index
- Year
- Month
- Quarter
- Previous month sales (`Lag_1`)
- Two-month lag (`Lag_2`)
- Three-month lag (`Lag_3`)
- Three-month rolling average

### 10. Model Evaluation

The models are evaluated using:

**MAE**

```text
MAE = average(|Actual - Predicted|)
```

**RMSE**

```text
RMSE = sqrt(average((Actual - Predicted)^2))
```

**R²**

```text
R² = 1 - SS_res / SS_tot
```

### 11. Future Forecast

The project forecasts the next **three months** using recursive regression forecasting.

---

## Descriptive Analysis

Descriptive analysis answers:

> **What happened?**

Examples:

- Which category generated the highest sales?
- Which region generated the highest sales?
- What is the average transaction sales value?
- Which products are top sellers?

---

## Diagnostic Analysis

Diagnostic analysis answers:

> **Why did it happen?**

Examples:

- Does discount relate to profit?
- Which categories generate high sales but lower profit?
- Which regions have strong sales performance?
- Is quantity associated with sales?

---

## Predictive Analysis

Predictive analysis answers:

> **What is likely to happen next?**

The project uses Linear Regression to estimate future monthly sales based on historical sales patterns and time/lag features.

---

## Why Regression?

Regression is appropriate because **Sales is a continuous numerical target**.

The model attempts to learn the relationship:

```text
Sales = f(Quantity, Discount, Category, Region, Time, Previous Sales)
```

For the monthly forecasting model, historical sales lags are particularly important because recent demand can provide information about near-future demand.

---

## Model Validation

For transaction-level regression, the dataset is divided into training and testing data.

For monthly forecasting, a **chronological split** is used rather than random splitting.

This is important because a forecasting model should not use future observations to predict earlier observations.

---

## Expected Business Value

The forecast can help a retail business:

- Plan inventory.
- Reduce over-stocking.
- Reduce stock-out risk.
- Improve purchasing decisions.
- Identify strong-selling products/categories.
- Prepare for changes in monthly demand.
- Support sales planning.

---

## Business Interpretation

If forecasted sales increase:

```text
Higher expected demand
        ↓
Review inventory levels
        ↓
Increase replenishment where appropriate
        ↓
Reduce stock-out risk
```

If forecasted sales decrease:

```text
Lower expected demand
        ↓
Review existing inventory
        ↓
Avoid unnecessary purchasing
        ↓
Reduce over-stocking cost
```

Forecasts should not be used alone. Promotions, holidays, seasonality, supplier lead time, pricing, and business knowledge should also be considered.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Place the CSV file in the same directory as the Python program:

```text
stores_sales_forecasting.csv
```

Then run:

```bash
python retail_sales_forecasting.py
```

Or copy the sections into a Jupyter Notebook and execute them step by step.

---

## Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Project Workflow

```text
Data Loading
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
Descriptive Analysis
     ↓
Diagnostic Analysis
     ↓
Data Visualization
     ↓
Correlation Analysis
     ↓
Feature Preparation
     ↓
Transaction-Level Regression
     ↓
Monthly Forecasting Regression
     ↓
Model Evaluation
     ↓
Future Sales Forecast
     ↓
Product Analysis
     ↓
Business Insights
```

---

## Conclusion

This project demonstrates an end-to-end retail sales analytics workflow.

It combines **descriptive analysis, diagnostic analysis, visualization, correlation analysis, regression, predictive forecasting, and business interpretation**.

The final sales forecast can be used as a planning input to help balance inventory availability against the cost of holding excess stock.
