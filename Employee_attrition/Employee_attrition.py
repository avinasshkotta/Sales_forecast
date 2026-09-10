# ==============================================================================
# HR TECH - EMPLOYEE ATTRITION PREDICTION
# Classification Project
#
# Dataset:
# WA_Fn-UseC_-HR-Employee-Attrition.csv
#
# Objective:
# Predict whether an employee is likely to leave the organization.
#
# Analytics:
# 1. Data Loading
# 2. Data Understanding
# 3. Data Cleaning
# 4. Descriptive Analysis
# 5. Diagnostic Analysis
# 6. Data Visualization
# 7. Correlation Analysis
# 8. Feature Preparation
# 9. Machine Learning
# 10. Model Evaluation
# 11. Attrition Probability Prediction
# 12. Business Insights
# ==============================================================================


# ==============================================================================
# 0. IMPORT REQUIRED LIBRARIES
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

print("=" * 80)
print("1. DATA LOADING")
print("=" * 80)

file_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Shape:", df.shape)

print("\nFirst 5 records:")
print(df.head())


# ==============================================================================
# 2. DATA UNDERSTANDING
# ==============================================================================

print("\n" + "=" * 80)
print("2. DATA UNDERSTANDING")
print("=" * 80)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Variable Distribution:")
print(df["Attrition"].value_counts())

print("\nTarget Variable Percentage:")
print(df["Attrition"].value_counts(normalize=True) * 100)


# ==============================================================================
# 3. DATA CLEANING
# ==============================================================================

print("\n" + "=" * 80)
print("3. DATA CLEANING")
print("=" * 80)

# Remove duplicate records
df = df.drop_duplicates()

# Remove columns that contain constant or identifier information.
# These columns do not provide useful predictive information.
columns_to_drop = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(columns=columns_to_drop, errors="ignore")

# Check missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

print("\nDataset shape after cleaning:")
print(df.shape)

print("\nRemaining columns:")
print(df.columns.tolist())


# ==============================================================================
# 4. DESCRIPTIVE ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("4. DESCRIPTIVE ANALYSIS")
print("=" * 80)

# Numerical descriptive statistics
print("\nNumerical Descriptive Statistics:")
print(df.describe())

# Attrition count
print("\nAttrition Count:")
print(df["Attrition"].value_counts())

# Attrition percentage
attrition_percentage = (
    df["Attrition"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nAttrition Percentage:")
print(attrition_percentage)

# Important HR variables
important_numeric = [
    "Age",
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "WorkLifeBalance"
]

available_numeric = [
    col for col in important_numeric if col in df.columns
]

print("\nSelected Variable Statistics:")
print(df[available_numeric].describe())


# ==============================================================================
# 5. DIAGNOSTIC ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("5. DIAGNOSTIC ANALYSIS")
print("=" * 80)

# Attrition by overtime
print("\nAttrition by OverTime:")
print(
    pd.crosstab(
        df["OverTime"],
        df["Attrition"],
        normalize="index"
    ).round(3)
)

# Attrition by job role
print("\nAttrition by Job Role:")
print(
    pd.crosstab(
        df["JobRole"],
        df["Attrition"],
        normalize="index"
    ).round(3)
)

# Attrition by department
print("\nAttrition by Department:")
print(
    pd.crosstab(
        df["Department"],
        df["Attrition"],
        normalize="index"
    ).round(3)
)

# Attrition by business travel
print("\nAttrition by Business Travel:")
print(
    pd.crosstab(
        df["BusinessTravel"],
        df["Attrition"],
        normalize="index"
    ).round(3)
)

# Average values for employees who left vs stayed
diagnostic_columns = [
    "Age",
    "MonthlyIncome",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

diagnostic_columns = [
    col for col in diagnostic_columns if col in df.columns
]

print("\nAverage values by Attrition:")
print(
    df.groupby("Attrition")[diagnostic_columns]
    .mean()
    .round(2)
)


# ==============================================================================
# 6. DATA VISUALIZATION
# ==============================================================================

print("\n" + "=" * 80)
print("6. DATA VISUALIZATION")
print("=" * 80)


# ------------------------------------------------------------------------------
# 6.1 Attrition Distribution
# ------------------------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Attrition"
)

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 6.2 Attrition by Overtime
# ------------------------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="OverTime",
    hue="Attrition"
)

plt.title("Employee Attrition by Overtime")
plt.xlabel("Overtime")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 6.3 Attrition by Job Role
# ------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    y="JobRole",
    hue="Attrition"
)

plt.title("Employee Attrition by Job Role")
plt.xlabel("Number of Employees")
plt.ylabel("Job Role")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 6.4 Monthly Income vs Attrition
# ------------------------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome"
)

plt.title("Monthly Income vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 6.5 Years at Company vs Attrition
# ------------------------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Attrition",
    y="YearsAtCompany"
)

plt.title("Years at Company vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Years at Company")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 6.6 Job Satisfaction vs Attrition
# ------------------------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="JobSatisfaction",
    hue="Attrition"
)

plt.title("Job Satisfaction vs Employee Attrition")
plt.xlabel("Job Satisfaction")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 6.7 Age Distribution
# ------------------------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.histplot(
    data=df,
    x="Age",
    hue="Attrition",
    kde=True,
    bins=20
)

plt.title("Age Distribution by Attrition")
plt.xlabel("Age")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 6.8 Correlation Heatmap
# ------------------------------------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(14, 10))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title("Correlation Heatmap of Numerical Variables")

plt.tight_layout()
plt.show()


# ==============================================================================
# 7. CORRELATION ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("7. CORRELATION ANALYSIS")
print("=" * 80)

correlation_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))

# Correlation with selected variables
print("\nCorrelation with Monthly Income:")

if "MonthlyIncome" in correlation_matrix.columns:
    print(
        correlation_matrix["MonthlyIncome"]
        .sort_values(ascending=False)
        .round(3)
    )

print("\nCorrelation with YearsAtCompany:")

if "YearsAtCompany" in correlation_matrix.columns:
    print(
        correlation_matrix["YearsAtCompany"]
        .sort_values(ascending=False)
        .round(3)
    )


# ==============================================================================
# 8. FEATURE PREPARATION
# ==============================================================================

print("\n" + "=" * 80)
print("8. FEATURE PREPARATION")
print("=" * 80)

# Convert target variable to binary
df["Attrition_Target"] = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})

# Remove original target
X = df.drop(
    columns=["Attrition", "Attrition_Target"]
)

y = df["Attrition_Target"]

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)

print("\nTarget Distribution:")
print(y.value_counts())


# ==============================================================================
# 9. MACHINE LEARNING
# ==============================================================================

print("\n" + "=" * 80)
print("9. MACHINE LEARNING")
print("=" * 80)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])


# ------------------------------------------------------------------------------
# Preprocessing Pipeline
# ------------------------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ------------------------------------------------------------------------------
# Model 1: Logistic Regression
# ------------------------------------------------------------------------------

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_predictions = logistic_model.predict(X_test)

logistic_probabilities = logistic_model.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------------------------
# Model 2: Random Forest
# ------------------------------------------------------------------------------

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                max_depth=10
            )
        )
    ]
)

random_forest_model.fit(
    X_train,
    y_train
)

rf_predictions = random_forest_model.predict(X_test)

rf_probabilities = random_forest_model.predict_proba(
    X_test
)[:, 1]


# ==============================================================================
# 10. MODEL EVALUATION
# ==============================================================================

print("\n" + "=" * 80)
print("10. MODEL EVALUATION")
print("=" * 80)


def evaluate_model(
    model_name,
    y_true,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Stayed", "Left"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_true,
        predictions
    )

    print(cm)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_predictions,
    logistic_probabilities
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions,
    rf_probabilities
)


# ------------------------------------------------------------------------------
# Model Comparison
# ------------------------------------------------------------------------------

results_df = pd.DataFrame([
    logistic_results,
    rf_results
])

print("\nModel Comparison:")
print(
    results_df.round(4)
)


# ------------------------------------------------------------------------------
# Model Comparison Visualization
# ------------------------------------------------------------------------------

plt.figure(figsize=(10, 6))

results_melted = results_df.melt(
    id_vars="Model",
    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    var_name="Metric",
    value_name="Score"
)

sns.barplot(
    data=results_melted,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.title("Machine Learning Model Comparison")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# Confusion Matrix - Best Model
# ------------------------------------------------------------------------------

# Select model based on F1 score
best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(),
    "Model"
]

print("\nBest Model based on F1 Score:")
print(best_model_name)

if best_model_name == "Logistic Regression":
    best_predictions = logistic_predictions
    best_probabilities = logistic_probabilities
else:
    best_predictions = rf_predictions
    best_probabilities = rf_probabilities


cm = confusion_matrix(
    y_test,
    best_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Left"],
    yticklabels=["Stayed", "Left"]
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------------------------

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_probabilities
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression "
          f"(AUC = {roc_auc_score(y_test, logistic_probabilities):.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest "
          f"(AUC = {roc_auc_score(y_test, rf_probabilities):.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "k--"
)

plt.title("ROC Curve - Employee Attrition Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()
plt.show()


# ==============================================================================
# 11. ATTRITION PROBABILITY PREDICTION
# ==============================================================================

print("\n" + "=" * 80)
print("11. ATTRITION PROBABILITY PREDICTION")
print("=" * 80)

# Create a copy of test data
prediction_results = X_test.copy()

prediction_results["Actual_Attrition"] = y_test.values

prediction_results["Attrition_Probability"] = (
    best_probabilities * 100
).round(2)

prediction_results["Predicted_Attrition"] = np.where(
    best_probabilities >= 0.50,
    "Yes",
    "No"
)

# Risk category
prediction_results["Risk_Level"] = pd.cut(
    best_probabilities,
    bins=[-np.inf, 0.30, 0.60, np.inf],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)

print("\nEmployee Attrition Risk Predictions:")

columns_to_display = [
    col for col in [
        "Age",
        "JobRole",
        "Department",
        "MonthlyIncome",
        "JobSatisfaction",
        "OverTime",
        "YearsAtCompany",
        "Actual_Attrition",
        "Attrition_Probability",
        "Predicted_Attrition",
        "Risk_Level"
    ]
    if col in prediction_results.columns
]

print(
    prediction_results[
        columns_to_display
    ].head(20)
)


# ------------------------------------------------------------------------------
# High-Risk Employees
# ------------------------------------------------------------------------------

high_risk_employees = prediction_results[
    prediction_results["Risk_Level"] == "High Risk"
].copy()

high_risk_employees = high_risk_employees.sort_values(
    by="Attrition_Probability",
    ascending=False
)

print("\nNumber of High-Risk Employees:")
print(len(high_risk_employees))

print("\nTop High-Risk Employees:")

print(
    high_risk_employees[
        columns_to_display
    ].head(20)
)


# ------------------------------------------------------------------------------
# Risk Distribution Visualization
# ------------------------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=prediction_results,
    x="Risk_Level"
)

plt.title("Employee Attrition Risk Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# Attrition Probability Distribution
# ------------------------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.histplot(
    prediction_results["Attrition_Probability"],
    bins=20,
    kde=True
)

plt.title("Distribution of Employee Attrition Probability")
plt.xlabel("Attrition Probability (%)")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()


# ==============================================================================
# 12. BUSINESS INSIGHTS
# ==============================================================================

print("\n" + "=" * 80)
print("12. BUSINESS INSIGHTS")
print("=" * 80)

print("""
Key business questions answered by this project:

1. How many employees are leaving the organization?

2. Which job roles have higher attrition?

3. Does overtime appear to be associated with employee attrition?

4. Does job satisfaction differ between employees who stay and leave?

5. Does compensation influence attrition?

6. Does tenure influence attrition?

7. Which employee characteristics are associated with higher attrition?

8. Which machine learning model performs better?

9. Which employees have a high predicted probability of leaving?

10. How can HR use attrition predictions for proactive retention?
""")

print("\nProject completed successfully.")
