# 📊 Telecom Customer Churn Prediction

## 📌 Project Overview

Customer churn is one of the major challenges faced by telecom companies. **Customer churn** occurs when a customer cancels or discontinues their subscription or service.

The objective of this project is to analyze telecom customer data and build a machine learning classification model that predicts whether a customer is likely to churn.

Since acquiring a new customer can be more expensive than retaining an existing customer, identifying customers who are at high risk of churn can help the company take proactive retention measures.

This project implements:

- **Descriptive Analytics**
- **Diagnostic Analytics**
- **Predictive Analytics**
- **Data Visualization**
- **Machine Learning Classification**
- **Model Evaluation**
- **Feature Importance Analysis**
- **Customer Churn Risk Prediction**

---

## 🎯 Business Problem

### Problem Statement

> Predict which telecom customers are likely to cancel their subscription.

### Business Challenge

Customer churn can result in:

- Loss of recurring revenue
- Increased customer acquisition costs
- Reduced customer lifetime value
- Increased marketing expenses
- Loss of long-term customers

### Business Objective

The goal is to identify customers who are likely to churn **before they leave the company**, allowing the business to implement targeted customer-retention strategies.

---

# 📂 Dataset

The project uses the **Telco Customer Churn Dataset**.

### Dataset File

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Dataset Description

The dataset contains customer-level information related to:

- Demographics
- Customer tenure
- Phone services
- Internet services
- Contract information
- Payment methods
- Monthly charges
- Total charges
- Customer churn status

### Target Variable

```text
Churn
```

The target variable contains two classes:

| Value | Meaning |
|---|---|
| `Yes` | Customer has churned |
| `No` | Customer has not churned |

For machine learning, these values are converted into:

```text
0 → No Churn
1 → Churn
```

---

# 🛠️ Technologies Used

The project is implemented using **Python** and **Jupyter Notebook**.

### Programming Language

- Python

### Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Machine Learning Algorithms

- Logistic Regression
- Random Forest Classifier

---

# 📦 Project Structure

```text
Telecom-Customer-Churn/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── Telecom_Customer_Churn.ipynb
│
├── customer_churn_predictions.csv
│
└── README.md
```

---

# 🔄 Project Workflow

The project follows an end-to-end data analytics and machine learning workflow:

```text
                    TELECOM CUSTOMER DATA
                              |
                              ↓
                    Data Loading & Cleaning
                              |
                              ↓
                    Exploratory Data Analysis
                              |
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
        DESCRIPTIVE       DIAGNOSTIC       PREDICTIVE
         ANALYTICS         ANALYTICS        ANALYTICS
              |               |               |
              ↓               ↓               ↓
        What happened?    Why happened?   What may happen?
              |               |               |
              └───────────────┼───────────────┘
                              ↓
                       Data Visualization
                              |
                              ↓
                    Machine Learning Models
                              |
                    ┌─────────┴─────────┐
                    ↓                   ↓
             Logistic Regression   Random Forest
                    |                   |
                    └─────────┬─────────┘
                              ↓
                       Model Evaluation
                              |
                              ↓
                     Churn Probability
                              |
                              ↓
                    High-Risk Customers
                              |
                              ↓
                    Retention Strategies
```

---

# 📊 Analytics Performed

## 1. Descriptive Analytics

### Question:

> **What happened?**

Descriptive analytics summarizes the existing customer data and provides an understanding of the customer base.

### Steps Covered

**Steps 5–14**

The analysis includes:

- Target variable distribution
- Numerical summary statistics
- Categorical variable summaries
- Overall churn rate
- Churn distribution
- Gender distribution
- Senior citizen distribution
- Customer tenure distribution
- Monthly charges distribution
- Total charges distribution

### Key Metrics

The overall churn rate is calculated using:

\[
Churn\ Rate =
\frac{Number\ of\ Churned\ Customers}
{Total\ Number\ of\ Customers}
\times 100
\]

---

# 🔎 2. Diagnostic Analytics

### Question:

> **Why are customers churning?**

Diagnostic analytics investigates relationships between customer characteristics and churn.

### Steps Covered

**Steps 15–24**

The analysis includes:

- Churn by contract type
- Churn rate by contract
- Churn by internet service
- Churn by payment method
- Churn by senior citizen status
- Churn versus tenure
- Churn versus monthly charges
- Churn versus total charges
- Correlation analysis
- Churn rate by tenure group

This analysis helps identify customer segments that may have a higher risk of churn.

---

# 🤖 3. Predictive Analytics

### Question:

> **Which customers are likely to churn?**

Predictive analytics uses historical customer information to train machine learning models that predict future churn.

### Steps Covered

**Steps 25–38**

The predictive workflow includes:

1. Preparing the machine learning dataset
2. Encoding the target variable
3. Separating features and target
4. Identifying numerical and categorical features
5. Splitting data into training and testing sets
6. Standardizing numerical variables
7. One-hot encoding categorical variables
8. Training Logistic Regression
9. Training Random Forest
10. Evaluating model performance
11. Generating confusion matrices
12. Comparing models
13. Creating ROC curves
14. Analyzing feature importance
15. Predicting churn probability
16. Identifying high-risk customers

---

# 🧹 Data Preprocessing

The following preprocessing operations are performed before model training.

### Remove Customer ID

The `customerID` column is removed because it is an identifier and does not provide meaningful predictive information.

### Convert Total Charges

`TotalCharges` is converted from text to numeric format.

Invalid or blank values are converted to missing values and subsequently handled using the median.

### Target Encoding

The `Churn` column is converted into binary values:

```text
No  → 0
Yes → 1
```

---

# 🔢 Feature Preprocessing

The dataset contains both numerical and categorical variables.

### Numerical Features

Numerical features are standardized using:

```python
StandardScaler()
```

Standardization transforms the variables so that they have a comparable scale.

### Categorical Features

Categorical variables are converted into numerical representations using:

```python
OneHotEncoder()
```

### ColumnTransformer

A `ColumnTransformer` is used to apply the appropriate preprocessing technique to each type of feature.

---

# 🧠 Machine Learning Models

## 1. Logistic Regression

Logistic Regression is a classification algorithm used to estimate the probability that a customer will churn.

The model produces a probability between 0 and 1.

Conceptually:

\[
P(Y=1|X)=
\frac{1}{1+e^{-z}}
\]

where:

\[
z = \beta_0+\beta_1X_1+\beta_2X_2+\cdots+\beta_nX_n
\]

A probability threshold is then used to classify customers as churn or non-churn.

---

## 2. Random Forest Classifier

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees.

Each tree makes a prediction, and the forest combines these predictions to produce the final classification.

Random Forest is useful for this project because it can:

- Capture nonlinear relationships
- Handle different types of features
- Model feature interactions
- Provide feature importance scores

---

# 📈 Model Evaluation

The machine learning models are evaluated using several classification metrics.

## Accuracy

Accuracy measures the proportion of total predictions that are correct.

\[
Accuracy =
\frac{TP+TN}
{TP+TN+FP+FN}
\]

---

## Precision

Precision measures how many customers predicted as churners actually churned.

\[
Precision =
\frac{TP}
{TP+FP}
\]

---

## Recall

Recall measures how many actual churners were successfully identified by the model.

\[
Recall =
\frac{TP}
{TP+FN}
\]

Recall is particularly important in customer churn prediction because failing to identify a customer who is going to churn may result in losing that customer.

---

## F1 Score

F1 Score is the harmonic mean of precision and recall.

\[
F1 =
2\times
\frac{Precision\times Recall}
{Precision+Recall}
\]

---

## ROC-AUC

ROC-AUC measures the model's ability to distinguish between churned and non-churned customers across different classification thresholds.

A higher ROC-AUC generally indicates better discrimination between the two classes.

---

# 🔲 Confusion Matrix

A confusion matrix provides a detailed view of classification results.

```text
                    Predicted
                 No Churn   Churn
               -------------------
Actual No Churn |    TN    |  FP  |
               -------------------
Actual Churn   |    FN    |  TP  |
               -------------------
```

Where:

- **TN** = True Negative
- **FP** = False Positive
- **FN** = False Negative
- **TP** = True Positive

---

# 📉 ROC Curve

The ROC curve compares the:

- True Positive Rate
- False Positive Rate

for different classification thresholds.

The project plots ROC curves for both:

- Logistic Regression
- Random Forest

and compares their ROC-AUC scores.

---

# ⭐ Feature Importance

Random Forest provides feature importance scores.

These scores help identify which customer attributes are most useful for predicting churn.

The project extracts the top 15 important features and visualizes them using a bar chart.

This analysis can help business stakeholders understand which customer characteristics should receive greater attention.

---

# 🎯 Customer Churn Risk Prediction

The trained Random Forest model generates a churn probability for customers.

Example:

```text
Customer A → Churn Probability: 0.82 → High Risk
Customer B → Churn Probability: 0.67 → High Risk
Customer C → Churn Probability: 0.23 → Low Risk
Customer D → Churn Probability: 0.11 → Low Risk
```

The project uses a probability threshold of `0.50` for the basic risk classification:

```text
Probability >= 0.50 → High Risk
Probability <  0.50 → Low Risk
```

---

# 📊 Visualizations

The project includes multiple visualizations to support descriptive, diagnostic, and predictive analysis.

### Descriptive Visualizations

- Customer churn distribution
- Gender distribution
- Senior citizen distribution
- Tenure distribution
- Monthly charges distribution
- Total charges distribution

### Diagnostic Visualizations

- Churn by contract
- Churn by internet service
- Churn by payment method
- Churn by senior citizen status
- Tenure versus churn
- Monthly charges versus churn
- Total charges versus churn
- Correlation heatmap
- Churn rate by tenure group

### Predictive Visualizations

- Confusion matrix
- Model comparison
- ROC curve
- Random Forest feature importance
- Predicted churn risk distribution

---

# 💡 Business Insights

The analysis is designed to help a telecom company identify customer segments that may have a higher risk of churn.

Potential retention strategies include:

### 1. Target High-Risk Customers

Use predicted churn probability to prioritize customers who are most likely to leave.

### 2. Personalized Offers

Provide targeted discounts or customized plans to high-risk customers.

### 3. Contract Incentives

Encourage suitable customers to move from short-term contracts to longer-term plans.

### 4. Customer Support

Provide proactive support to customers showing signs of dissatisfaction or churn risk.

### 5. Loyalty Programs

Reward long-term customers and encourage continued subscriptions.

### 6. Service Improvement

Investigate service categories associated with higher churn and improve the customer experience.

---

# 📁 Output File

The project generates:

```text
customer_churn_predictions.csv
```

This file contains prediction information such as:

- Customer tenure
- Contract
- Monthly charges
- Total charges
- Actual churn
- Predicted churn
- Churn probability
- Churn risk classification

The output can be used to prioritize customer-retention campaigns.

---

# ▶️ How to Run the Project

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

## 2. Navigate to the Project Directory

```bash
cd Telecom-Customer-Churn
```

## 3. Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open the Notebook

Open:

```text
Telecom_Customer_Churn.ipynb
```

## 6. Run the Cells

Run the notebook cells sequentially from Step 1 through Step 39.

Make sure the CSV dataset is located in the same directory as the notebook.

---

# 📋 Project Steps Summary

| Step | Category | Description |
|---:|---|---|
| 1 | Setup | Import libraries |
| 2 | Data | Load dataset |
| 3 | Data | Understand dataset |
| 4 | Data Cleaning | Clean dataset |
| 5 | Descriptive | Target distribution |
| 6 | Descriptive | Numerical statistics |
| 7 | Descriptive | Categorical summary |
| 8 | Descriptive | Overall churn rate |
| 9 | Visualization | Churn distribution |
| 10 | Visualization | Gender distribution |
| 11 | Visualization | Senior citizen distribution |
| 12 | Visualization | Tenure distribution |
| 13 | Visualization | Monthly charges |
| 14 | Visualization | Total charges |
| 15 | Diagnostic | Churn by contract |
| 16 | Diagnostic | Churn rate by contract |
| 17 | Diagnostic | Churn by internet service |
| 18 | Diagnostic | Churn by payment method |
| 19 | Diagnostic | Churn by senior citizen |
| 20 | Diagnostic | Churn vs tenure |
| 21 | Diagnostic | Churn vs monthly charges |
| 22 | Diagnostic | Churn vs total charges |
| 23 | Diagnostic | Correlation analysis |
| 24 | Diagnostic | Churn by tenure group |
| 25 | Predictive | Prepare ML data |
| 26 | Predictive | Train-test split |
| 27 | Predictive | Preprocessing |
| 28 | Predictive | Logistic Regression |
| 29 | Predictive | Logistic Regression confusion matrix |
| 30 | Predictive | Random Forest |
| 31 | Predictive | Random Forest confusion matrix |
| 32 | Predictive | Model comparison |
| 33 | Visualization | Model comparison chart |
| 34 | Predictive | ROC curve |
| 35 | Predictive | Feature importance |
| 36 | Visualization | Feature importance chart |
| 37 | Predictive | Churn probability |
| 38 | Predictive | Risk distribution |
| 39 | Business | Final insights and output |

---

# 📌 Key Takeaways

This project demonstrates a complete **data analytics and machine learning pipeline** for telecom customer churn prediction.

The project progresses from:

```text
Raw Data
   ↓
Data Cleaning
   ↓
Descriptive Analytics
   ↓
Diagnostic Analytics
   ↓
Predictive Analytics
   ↓
Machine Learning
   ↓
Model Evaluation
   ↓
Churn Probability
   ↓
High-Risk Customer Identification
   ↓
Customer Retention Strategy
```

The final objective is not only to predict churn but also to transform those predictions into **actionable business decisions** that can improve customer retention and reduce revenue loss.

---

# 👨‍💻 Author

**Avinassh Kotta**

### Project

**Telecom Customer Churn Prediction**

### Domain

**Telecommunications / Customer Analytics / Machine Learning**

### Analytics Types

**Descriptive | Diagnostic | Predictive**

### Machine Learning

**Classification**