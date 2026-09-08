# ============================================================
# TELECOM CUSTOMER CHURN PREDICTION
# Descriptive + Diagnostic + Predictive Analytics
# ============================================================


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# ============================================================
# STEP 2: LOAD THE DATASET
# ============================================================

file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("=" * 70)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)

print("Number of Rows    :", df.shape[0])
print("Number of Columns :", df.shape[1])

print("\nFirst 5 Records:")
display(df.head())


# ============================================================
# STEP 3: BASIC DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

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


# ============================================================
# STEP 4: DATA CLEANING
# ============================================================

# Remove unnecessary customer ID
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric
# Some records contain blank spaces
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values created by conversion
print("\nMissing values after converting TotalCharges:")
print(df.isnull().sum())

# Fill missing TotalCharges using median
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())


# ============================================================
# STEP 5: TARGET VARIABLE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET VARIABLE - CHURN")
print("=" * 70)

print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(
    df["Churn"].value_counts(normalize=True) * 100
)


# ============================================================
# DESCRIPTIVE ANALYTICS
# ============================================================

print("\n\n" + "=" * 70)
print("DESCRIPTIVE ANALYTICS")
print("=" * 70)


# ============================================================
# STEP 6: NUMERICAL SUMMARY
# ============================================================

print("\nNumerical Summary Statistics:")

display(
    df.describe()
)


# ============================================================
# STEP 7: CATEGORICAL SUMMARY
# ============================================================

categorical_columns = df.select_dtypes(
    include="object"
).columns

print("\nCategorical Variables:")

for column in categorical_columns:
    print("\n------------------------------------------")
    print(column)
    print("------------------------------------------")
    print(df[column].value_counts())


# ============================================================
# STEP 8: OVERALL CHURN RATE
# ============================================================

churn_count = df["Churn"].value_counts()

print("\nTotal Customers :", len(df))
print("Customers Churned :", churn_count["Yes"])
print("Customers Retained:", churn_count["No"])

churn_rate = (
    df["Churn"].value_counts(normalize=True)["Yes"] * 100
)

print(f"Overall Churn Rate: {churn_rate:.2f}%")


# ============================================================
# STEP 9: DESCRIPTIVE VISUALIZATION
# CHURN DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 10: GENDER DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="gender"
)

plt.title("Customer Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 11: SENIOR CITIZEN DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="SeniorCitizen"
)

plt.title("Senior Citizen Distribution")
plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 12: TENURE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="tenure",
    bins=30,
    kde=True
)

plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 13: MONTHLY CHARGES DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="MonthlyCharges",
    bins=30,
    kde=True
)

plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 14: TOTAL CHARGES DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="TotalCharges",
    bins=30,
    kde=True
)

plt.title("Total Charges Distribution")
plt.xlabel("Total Charges")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# DIAGNOSTIC ANALYTICS
# ============================================================

print("\n\n" + "=" * 70)
print("DIAGNOSTIC ANALYTICS")
print("=" * 70)

print("""
Diagnostic Analytics answers:

"WHY are customers churning?"

We compare Churn against important customer characteristics
such as contract, tenure, internet service, payment method,
monthly charges and senior citizen status.
""")


# ============================================================
# STEP 15: CHURN BY CONTRACT
# ============================================================

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Contract:")
display(contract_churn)


plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=10)

plt.tight_layout()
plt.show()


# ============================================================
# STEP 16: CHURN RATE BY CONTRACT
# ============================================================

contract_churn_plot = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="ChurnRate")
)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=contract_churn_plot,
    x="Contract",
    y="ChurnRate"
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 17: CHURN BY INTERNET SERVICE
# ============================================================

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Internet Service:")
display(internet_churn)


plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="InternetService",
    hue="Churn"
)

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 18: CHURN BY PAYMENT METHOD
# ============================================================

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Payment Method:")
display(payment_churn)


plt.figure(figsize=(11, 6))

sns.countplot(
    data=df,
    x="PaymentMethod",
    hue="Churn"
)

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# ============================================================
# STEP 19: CHURN BY SENIOR CITIZEN
# ============================================================

senior_churn = pd.crosstab(
    df["SeniorCitizen"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Senior Citizen Status:")
display(senior_churn)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="SeniorCitizen",
    hue="Churn"
)

plt.title("Customer Churn by Senior Citizen Status")
plt.xlabel("Senior Citizen (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 20: CHURN VS TENURE
# ============================================================

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure Distribution by Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 21: CHURN VS MONTHLY CHARGES
# ============================================================

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 22: CHURN VS TOTAL CHARGES
# ============================================================

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="TotalCharges"
)

plt.title("Total Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Total Charges")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 23: CORRELATION ANALYSIS
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
)

correlation_matrix = numeric_columns.corr()

print("\nCorrelation Matrix:")
display(correlation_matrix)


plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 24: CHURN RATE BY TENURE GROUP
# ============================================================

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 60, 100],
    labels=[
        "0-12 Months",
        "13-24 Months",
        "25-48 Months",
        "49-60 Months",
        "60+ Months"
    ]
)

tenure_churn = (
    df.groupby("TenureGroup", observed=False)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="ChurnRate")
)

print("\nChurn Rate by Tenure Group:")
display(tenure_churn)


plt.figure(figsize=(10, 5))

sns.barplot(
    data=tenure_churn,
    x="TenureGroup",
    y="ChurnRate"
)

plt.title("Churn Rate by Customer Tenure")
plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()
plt.show()


# ============================================================
# PREDICTIVE ANALYTICS
# ============================================================

print("\n\n" + "=" * 70)
print("PREDICTIVE ANALYTICS")
print("=" * 70)

print("""
Predictive Analytics answers:

"Which customers are likely to churn?"

We will train machine learning models to predict
whether a customer will churn.

Models:
1. Logistic Regression
2. Random Forest Classifier
""")


# ============================================================
# STEP 25: PREPARE DATA FOR MACHINE LEARNING
# ============================================================

# Remove helper column
df_model = df.drop("TenureGroup", axis=1)

# Convert target variable
df_model["Churn"] = df_model["Churn"].map({
    "No": 0,
    "Yes": 1
})


# Separate features and target
X = df_model.drop("Churn", axis=1)
y = df_model["Churn"]


# Identify categorical and numerical columns
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# STEP 26: TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", X_train.shape[0])
print("Testing Records :", X_test.shape[0])


# ============================================================
# STEP 27: PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
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


# ============================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# Train model
logistic_model.fit(
    X_train,
    y_train
)


# Make predictions
y_pred_lr = logistic_model.predict(X_test)

y_prob_lr = logistic_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# STEP 28: LOGISTIC REGRESSION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 70)

lr_accuracy = accuracy_score(
    y_test,
    y_pred_lr
)

lr_precision = precision_score(
    y_test,
    y_pred_lr
)

lr_recall = recall_score(
    y_test,
    y_pred_lr
)

lr_f1 = f1_score(
    y_test,
    y_pred_lr
)

lr_auc = roc_auc_score(
    y_test,
    y_prob_lr
)

print(f"Accuracy  : {lr_accuracy:.4f}")
print(f"Precision : {lr_precision:.4f}")
print(f"Recall    : {lr_recall:.4f}")
print(f"F1 Score  : {lr_f1:.4f}")
print(f"ROC-AUC   : {lr_auc:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_lr
    )
)


# ============================================================
# STEP 29: LOGISTIC REGRESSION CONFUSION MATRIX
# ============================================================

cm_lr = confusion_matrix(
    y_test,
    y_pred_lr
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# MODEL 2: RANDOM FOREST
# ============================================================

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
                n_jobs=-1
            )
        )
    ]
)


# Train model
random_forest_model.fit(
    X_train,
    y_train
)


# Predictions
y_pred_rf = random_forest_model.predict(
    X_test
)

y_prob_rf = random_forest_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# STEP 30: RANDOM FOREST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

rf_precision = precision_score(
    y_test,
    y_pred_rf
)

rf_recall = recall_score(
    y_test,
    y_pred_rf
)

rf_f1 = f1_score(
    y_test,
    y_pred_rf
)

rf_auc = roc_auc_score(
    y_test,
    y_prob_rf
)

print(f"Accuracy  : {rf_accuracy:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1 Score  : {rf_f1:.4f}")
print(f"ROC-AUC   : {rf_auc:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_rf
    )
)


# ============================================================
# STEP 31: RANDOM FOREST CONFUSION MATRIX
# ============================================================

cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 32: MODEL COMPARISON
# ============================================================

model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        lr_accuracy,
        rf_accuracy
    ],
    "Precision": [
        lr_precision,
        rf_precision
    ],
    "Recall": [
        lr_recall,
        rf_recall
    ],
    "F1 Score": [
        lr_f1,
        rf_f1
    ],
    "ROC-AUC": [
        lr_auc,
        rf_auc
    ]
})

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

display(model_comparison)


# ============================================================
# STEP 33: MODEL COMPARISON VISUALIZATION
# ============================================================

comparison_melted = model_comparison.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=comparison_melted,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Evaluation Metric")
plt.ylabel("Score")

plt.ylim(0, 1)

plt.tight_layout()
plt.show()


# ============================================================
# STEP 34: ROC CURVE
# ============================================================

fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    y_prob_lr
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    y_prob_rf
)


plt.figure(figsize=(8, 6))

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression (AUC = {lr_auc:.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {rf_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title("ROC Curve - Customer Churn Prediction")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# STEP 35: RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

# Get trained preprocessing object
rf_preprocessor = (
    random_forest_model
    .named_steps["preprocessor"]
)

rf_classifier = (
    random_forest_model
    .named_steps["classifier"]
)


# Get transformed feature names
feature_names = (
    rf_preprocessor
    .get_feature_names_out()
)


# Get importance values
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_classifier.feature_importances_
})


# Sort
feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


print("\n" + "=" * 70)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 70)

display(
    feature_importance.head(20)
)


# ============================================================
# STEP 36: FEATURE IMPORTANCE VISUALIZATION
# ============================================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Features Influencing Customer Churn"
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 37: PREDICT CHURN PROBABILITY
# ============================================================

# Create prediction output
prediction_results = X_test.copy()

prediction_results["Actual_Churn"] = y_test.values

prediction_results["Predicted_Churn"] = y_pred_rf

prediction_results["Churn_Probability"] = y_prob_rf

prediction_results["Churn_Prediction"] = np.where(
    prediction_results["Churn_Probability"] >= 0.5,
    "High Risk",
    "Low Risk"
)


# Sort by highest churn probability
prediction_results = prediction_results.sort_values(
    "Churn_Probability",
    ascending=False
)


print("\n" + "=" * 70)
print("CUSTOMERS WITH HIGHEST CHURN RISK")
print("=" * 70)

display(
    prediction_results[
        [
            "tenure",
            "Contract",
            "MonthlyCharges",
            "TotalCharges",
            "Churn_Probability",
            "Churn_Prediction"
        ]
    ].head(20)
)


# ============================================================
# STEP 38: CHURN RISK DISTRIBUTION
# ============================================================

risk_distribution = (
    prediction_results["Churn_Prediction"]
    .value_counts()
    .reset_index()
)

risk_distribution.columns = [
    "RiskLevel",
    "CustomerCount"
]

print("\nCustomer Churn Risk Distribution:")
display(risk_distribution)


plt.figure(figsize=(7, 5))

sns.countplot(
    data=prediction_results,
    x="Churn_Prediction"
)

plt.title("Predicted Customer Churn Risk")
plt.xlabel("Risk Level")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 39: FINAL BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL BUSINESS INSIGHTS")
print("=" * 70)

print("""
1. DESCRIPTIVE ANALYTICS
------------------------
Descriptive analytics identifies the overall customer base,
customer characteristics and the percentage of customers
who have already churned.

2. DIAGNOSTIC ANALYTICS
-----------------------
Diagnostic analytics identifies factors associated with churn,
including contract type, tenure, monthly charges, internet
service and payment method.

3. PREDICTIVE ANALYTICS
-----------------------
Machine learning models predict customers who are likely
to cancel their subscription.

4. BUSINESS ACTION
------------------
Customers predicted as high-risk can be targeted with
retention strategies such as:

- Personalized discounts
- Contract upgrade offers
- Better service plans
- Customer support intervention
- Loyalty rewards
- Customized pricing
- Long-term contract incentives

5. BUSINESS OBJECTIVE
---------------------
The ultimate goal is to identify high-risk customers early
and take preventive action before they leave the company.
""")


# ============================================================
# STEP 40: SAVE PREDICTIONS
# ============================================================

prediction_results.to_csv(
    "customer_churn_predictions.csv",
    index=False
)

print("\nPrediction file saved as:")
print("customer_churn_predictions.csv")


# ============================================================
# END OF ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER CHURN ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)