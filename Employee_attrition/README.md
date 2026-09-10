```markdown
# HR Tech – Employee Attrition Prediction

## 📌 Project Overview

Employee attrition is an important business problem for organizations because losing experienced employees can result in recruitment costs, training expenses, productivity loss, and disruption to teams.

This project uses **Machine Learning Classification** to predict whether an employee is likely to leave the organization based on employee characteristics such as:

- Job Role
- Job Satisfaction
- Environment Satisfaction
- Monthly Income
- Overtime
- Age
- Total Working Years
- Years at Company
- Years in Current Role
- Years Since Last Promotion
- Work-Life Balance
- Business Travel
- Department

The project follows an end-to-end **HR Analytics and Machine Learning workflow**.

---

## 🎯 Business Problem

### Problem

Organizations need to identify employees who may be at risk of leaving so that HR teams can take proactive retention measures.

### Business Impact

Employee turnover can lead to:

- Recruitment costs
- Training costs
- Loss of experienced employees
- Productivity reduction
- Team disruption
- Knowledge loss
- Increased workload for existing employees

### Proposed Solution

Build a machine learning classification model that predicts employee attrition and generates an estimated probability of an employee leaving.

This allows HR teams to identify potentially high-risk employees and consider appropriate retention strategies.

---

# 📊 Dataset

The project uses the **IBM HR Analytics Employee Attrition & Performance** dataset.

### Dataset File

```text
WA_Fn-UseC_-HR-Employee-Attrition.csv
```

### Dataset Size

The dataset contains:

- **1,470 employee records**
- **35 original columns**

The target variable is:

```text
Attrition
```

Target classes:

```text
Yes = Employee left
No  = Employee stayed
```

---

# 🔄 Project Workflow

The project is divided into the following stages:

```text
1. Data Loading
        ↓
2. Data Understanding
        ↓
3. Data Cleaning
        ↓
4. Descriptive Analysis
        ↓
5. Diagnostic Analysis
        ↓
6. Data Visualization
        ↓
7. Correlation Analysis
        ↓
8. Feature Preparation
        ↓
9. Machine Learning
        ↓
10. Model Evaluation
        ↓
11. Attrition Probability Prediction
        ↓
12. Business Insights
        ↓
13. HR Retention Recommendations
        ↓
14. Conclusion
```

---

# 1️⃣ Data Loading

The employee attrition dataset is loaded using Pandas.

```python
df = pd.read_csv(
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)
```

The dataset is then inspected to verify that it has been loaded correctly.

---

# 2️⃣ Data Understanding

The dataset is explored using:

- `head()`
- `shape`
- `columns`
- `dtypes`
- `info()`
- `describe()`
- Missing-value analysis
- Duplicate analysis
- Target-variable analysis

This stage helps understand the structure and quality of the dataset.

---

# 3️⃣ Data Cleaning

The dataset is checked for:

- Missing values
- Duplicate records
- Unnecessary columns
- Identifier columns
- Constant columns

The following columns are removed because they do not provide useful predictive information:

```text
EmployeeCount
EmployeeNumber
Over18
StandardHours
```

---

# 4️⃣ Descriptive Analysis

Descriptive analytics summarizes the employee dataset.

The following statistics are analyzed:

- Mean
- Median
- Standard deviation
- Minimum
- Maximum
- Quartiles
- Employee attrition percentage

Important HR variables include:

- Age
- Monthly Income
- Total Working Years
- Years at Company
- Job Satisfaction
- Environment Satisfaction
- Job Involvement
- Work-Life Balance

---

# 5️⃣ Diagnostic Analysis

Diagnostic analysis attempts to understand factors associated with employee attrition.

The project examines:

### Attrition vs Overtime

Determines whether employees working overtime have different attrition rates.

### Attrition vs Job Role

Identifies job roles with relatively higher attrition.

### Attrition vs Department

Compares attrition across organizational departments.

### Attrition vs Business Travel

Examines whether travel frequency is associated with employee attrition.

### Attrition vs Satisfaction

Compares satisfaction levels between employees who stayed and employees who left.

### Attrition vs Compensation

Examines monthly income differences between attrition groups.

### Attrition vs Tenure

Examines the relationship between years at the company and attrition.

---

# 6️⃣ Data Visualization

Several visualizations are generated to understand employee attrition patterns.

### Visualizations include:

- Employee Attrition Distribution
- Attrition by Overtime
- Attrition by Job Role
- Monthly Income vs Attrition
- Years at Company vs Attrition
- Job Satisfaction vs Attrition
- Age Distribution
- Correlation Heatmap
- Model Comparison
- Confusion Matrix
- ROC Curve
- Attrition Risk Distribution
- Attrition Probability Distribution

---

# 7️⃣ Correlation Analysis

Correlation analysis is performed on numerical variables.

The correlation matrix helps identify relationships between variables such as:

- Age
- Monthly Income
- Total Working Years
- Years at Company
- Years in Current Role
- Years Since Last Promotion
- Years With Current Manager
- Job Satisfaction
- Job Involvement
- Work-Life Balance

A heatmap is generated to visualize these relationships.

---

# 8️⃣ Feature Preparation

The target variable is converted into binary values:

```text
No  → 0
Yes → 1
```

The features are divided into:

### Numerical Features

Examples:

```text
Age
MonthlyIncome
DailyRate
HourlyRate
TotalWorkingYears
YearsAtCompany
YearsInCurrentRole
YearsSinceLastPromotion
YearsWithCurrManager
```

### Categorical Features

Examples:

```text
BusinessTravel
Department
EducationField
JobRole
MaritalStatus
OverTime
```

Numerical features are standardized using:

```python
StandardScaler
```

Categorical features are encoded using:

```python
OneHotEncoder
```

---

# 9️⃣ Machine Learning

This project uses two classification algorithms.

## Logistic Regression

Logistic Regression predicts the probability that an employee belongs to the attrition class.

Advantages:

- Simple
- Interpretable
- Provides probability estimates
- Suitable for binary classification

---

## Random Forest

Random Forest combines multiple decision trees to make predictions.

Advantages:

- Handles nonlinear relationships
- Works with many features
- Captures feature interactions
- Generally robust for tabular datasets

---

# 🔟 Model Evaluation

The models are evaluated using:

### Accuracy

Measures the percentage of total predictions that are correct.

### Precision

Measures how many employees predicted to leave actually leave.

### Recall

Measures how many employees who actually leave are correctly identified.

### F1 Score

Combines precision and recall into a single metric.

### ROC-AUC

Measures how effectively the model separates employees who leave from employees who stay.

---

# 📈 Model Comparison

The project compares:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | Calculated by program | Calculated by program | Calculated by program | Calculated by program | Calculated by program |
| Random Forest | Calculated by program | Calculated by program | Calculated by program | Calculated by program | Calculated by program |

The Python program automatically determines the best model based on **F1 Score**.

---

# 1️⃣1️⃣ Attrition Probability Prediction

The selected model generates an attrition probability for each employee.

For example:

```text
Employee A → 12% → Low Risk
Employee B → 47% → Medium Risk
Employee C → 82% → High Risk
```

The project categorizes employees into:

| Probability | Risk |
|---|---|
| Below 30% | Low Risk |
| 30% – 60% | Medium Risk |
| Above 60% | High Risk |

High-risk employees can be prioritized for further HR review.

---

# 🧠 Business Insights

The project helps HR teams answer questions such as:

- Which employees are most likely to leave?
- Which job roles have higher attrition?
- Is overtime associated with attrition?
- Does job satisfaction affect attrition?
- Does compensation influence attrition?
- Does tenure affect attrition?
- Which employee characteristics are associated with turnover?
- Which machine learning model performs better?
- Which employees have a high predicted probability of leaving?

---

# 💼 HR Retention Strategies

Based on the analysis, organizations can consider:

### 1. Retention Conversations

Conduct proactive discussions with employees identified as potentially high risk.

### 2. Overtime Management

Review excessive overtime and workload distribution.

### 3. Career Development

Provide career progression and professional development opportunities.

### 4. Compensation Review

Review compensation competitiveness where appropriate.

### 5. Employee Satisfaction

Address workplace factors associated with low satisfaction.

### 6. Work-Life Balance

Introduce measures that improve employee work-life balance.

### 7. Role-Specific Interventions

Develop targeted retention strategies for roles or departments with elevated attrition.

> **Important:** Machine learning predictions should be treated as decision-support information and not as automatic decisions about individual employees.

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning

- Logistic Regression
- Random Forest Classifier

## Data Processing

- StandardScaler
- OneHotEncoder
- ColumnTransformer
- Pipeline

## Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

---

# 📁 Project Structure

```text
HR_Employee_Attrition/
│
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── employee_attrition_prediction.py
│
├── HR_Employee_Attrition_Analysis.ipynb
│
└── README.md
```

---

# ▶️ How to Run the Project

## Step 1 – Clone the Repository

```bash
git clone <your-github-repository-url>
```

## Step 2 – Open the Project

Open the project using:

- PyCharm
- Jupyter Notebook
- VS Code

## Step 3 – Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Step 4 – Place the Dataset

Make sure the CSV file is located in the same directory as the Python program:

```text
WA_Fn-UseC_-HR-Employee-Attrition.csv
```

## Step 5 – Run the Program

```bash
python employee_attrition_prediction.py
```

---

# 📊 Expected Outputs

The program produces:

- Dataset summary
- Missing-value analysis
- Duplicate analysis
- Descriptive statistics
- Attrition percentages
- Diagnostic analysis
- Correlation matrix
- Multiple visualizations
- Logistic Regression model
- Random Forest model
- Model comparison
- Classification reports
- Confusion matrix
- ROC curves
- Attrition probabilities
- Employee risk categories
- High-risk employee list
- Business insights

---

# 🎯 Project Outcome

The final system provides an end-to-end solution for **employee attrition prediction**.

The project combines HR analytics with machine learning to identify potential attrition risks and provide actionable insights that can support employee retention strategies.

---

# 👨‍💻 Project Category

**Domain:** HR Technology / Human Resources Analytics

**Problem Type:** Classification

**Analytics Types:**

- Descriptive Analytics
- Diagnostic Analytics
- Predictive Analytics

**Machine Learning Type:** Supervised Learning

**Target:** Employee Attrition

**Models:**

- Logistic Regression
- Random Forest Classifier

---

# 📌 Conclusion

Employee attrition prediction can help organizations move from reactive employee management toward proactive retention planning.

By analyzing employee characteristics such as satisfaction, compensation, overtime, role, and tenure, machine learning can identify patterns associated with employee turnover.

The final objective is not simply to predict who may leave, but to provide HR teams with useful information that can support appropriate and timely retention initiatives.
```
