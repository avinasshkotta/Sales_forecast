# ==============================================================================
# CREDIT CARD FRAUD DETECTION
# Banking · Classification
#
# Dataset:
# Kaggle - Credit Card Fraud Detection
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#
# Dataset:
# 284,807 transactions
# 492 fraud cases
# 0.172% fraud rate
#
# Analytics:
# 1. Descriptive Analytics
# 2. Diagnostic Analytics
# 3. Predictive Analytics
#
# Machine Learning Models:
# 1. Logistic Regression
# 2. Random Forest
#
# Class Imbalance:
# SMOTE
#
# Evaluation:
# Accuracy
# Precision
# Recall
# F1 Score
# ROC-AUC
# PR-AUC
# Confusion Matrix
# ROC Curve
# Precision-Recall Curve
# ==============================================================================


# ==============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE


print("=" * 80)
print("CREDIT CARD FRAUD DETECTION")
print("=" * 80)
print("All libraries imported successfully.")


# ==============================================================================
# STEP 2: LOAD DATASET
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 2: LOAD DATASET")
print("=" * 80)

file_path = "creditcard.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully.")

print("\nDataset Shape:")
print(df.shape)

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nFirst 5 Rows:")
display(df.head())


# ==============================================================================
# STEP 3: BASIC DATA UNDERSTANDING
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 3: BASIC DATA UNDERSTANDING")
print("=" * 80)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()


# ==============================================================================
# STEP 4: CHECK MISSING VALUES
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 4: MISSING VALUE ANALYSIS")
print("=" * 80)

missing_values = df.isnull().sum()

print("\nMissing Values:")
display(missing_values.to_frame("Missing Values"))

print("\nTotal Missing Values:")
print(missing_values.sum())


# ==============================================================================
# STEP 5: CHECK DUPLICATE RECORDS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 5: DUPLICATE RECORD ANALYSIS")
print("=" * 80)

duplicate_count = df.duplicated().sum()

print("Number of Duplicate Rows:", duplicate_count)


# ==============================================================================
# STEP 6: REMOVE DUPLICATES
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 6: REMOVE DUPLICATES")
print("=" * 80)

if duplicate_count > 0:

    df = df.drop_duplicates().reset_index(drop=True)

    print("Duplicates removed.")

else:

    print("No duplicate records found.")

print("\nDataset Shape After Duplicate Handling:")
print(df.shape)


# ==============================================================================
# STEP 7: STATISTICAL SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 7: STATISTICAL SUMMARY")
print("=" * 80)

display(df.describe().T)


# ==============================================================================
# STEP 8: TARGET VARIABLE DISTRIBUTION
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 8: TARGET VARIABLE DISTRIBUTION")
print("=" * 80)

class_counts = df["Class"].value_counts()

print("\nClass Counts:")
print(class_counts)

print("\nClass Meaning:")
print("0 = Genuine Transaction")
print("1 = Fraudulent Transaction")


# ==============================================================================
# STEP 9: TARGET VARIABLE PERCENTAGE
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 9: TARGET VARIABLE PERCENTAGE")
print("=" * 80)

class_percentage = (
    df["Class"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nClass Percentage:")
print(class_percentage)

fraud_count = (df["Class"] == 1).sum()
genuine_count = (df["Class"] == 0).sum()
total_transactions = len(df)

fraud_percentage = (
    fraud_count / total_transactions
) * 100

print("\nTotal Transactions:", total_transactions)
print("Genuine Transactions:", genuine_count)
print("Fraudulent Transactions:", fraud_count)
print(f"Fraud Percentage: {fraud_percentage:.4f}%")


# ==============================================================================
# STEP 10: FRAUD VS GENUINE VISUALIZATION
# DESCRIPTIVE ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 10: FRAUD VS GENUINE TRANSACTIONS")
print("=" * 80)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Genuine vs Fraudulent Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.xticks(
    [0, 1],
    ["Genuine", "Fraud"]
)

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 11: TRANSACTION AMOUNT DISTRIBUTION
# DESCRIPTIVE ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 11: TRANSACTION AMOUNT DISTRIBUTION")
print("=" * 80)

print("\nAmount Statistics:")

display(
    df["Amount"].describe()
)

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Amount",
    bins=100,
    kde=True
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 12: TRANSACTION TIME DISTRIBUTION
# DESCRIPTIVE ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 12: TRANSACTION TIME DISTRIBUTION")
print("=" * 80)

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Time",
    bins=100
)

plt.title("Transaction Time Distribution")
plt.xlabel("Time")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 13: FRAUD VS GENUINE TRANSACTION AMOUNT
# DESCRIPTIVE ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 13: AMOUNT COMPARISON")
print("=" * 80)

amount_summary = (
    df.groupby("Class")["Amount"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    )
)

amount_summary.index = [
    "Genuine",
    "Fraud"
]

display(amount_summary)


# ==============================================================================
# STEP 14: BOX PLOT OF TRANSACTION AMOUNT
# DIAGNOSTIC ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 14: TRANSACTION AMOUNT BY CLASS")
print("=" * 80)

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Transaction Amount: Genuine vs Fraud")
plt.xlabel("Transaction Type")
plt.ylabel("Amount")

plt.xticks(
    [0, 1],
    ["Genuine", "Fraud"]
)

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 15: FRAUD TRANSACTION AMOUNT DISTRIBUTION
# DIAGNOSTIC ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 15: FRAUDULENT TRANSACTION AMOUNT DISTRIBUTION")
print("=" * 80)

fraud_transactions = df[df["Class"] == 1]

plt.figure(figsize=(10, 6))

sns.histplot(
    fraud_transactions["Amount"],
    bins=50,
    kde=True
)

plt.title("Fraudulent Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 16: GENUINE TRANSACTION AMOUNT DISTRIBUTION
# DIAGNOSTIC ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 16: GENUINE TRANSACTION AMOUNT DISTRIBUTION")
print("=" * 80)

genuine_transactions = df[df["Class"] == 0]

plt.figure(figsize=(10, 6))

sns.histplot(
    genuine_transactions["Amount"],
    bins=100,
    kde=True
)

plt.title("Genuine Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 17: FEATURE CORRELATION MATRIX
# DIAGNOSTIC ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 17: CORRELATION MATRIX")
print("=" * 80)

correlation_matrix = df.corr()

plt.figure(figsize=(18, 14))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 18: CORRELATION WITH FRAUD
# DIAGNOSTIC ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 18: FEATURES CORRELATED WITH FRAUD")
print("=" * 80)

target_correlation = (
    df.corr()["Class"]
    .drop("Class")
    .sort_values(
        key=lambda x: abs(x),
        ascending=False
    )
)

print("\nTop Features Correlated With Fraud:")

display(
    target_correlation.head(15)
)


# ==============================================================================
# STEP 19: VISUALIZE TOP FRAUD CORRELATIONS
# DIAGNOSTIC ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 19: TOP FRAUD CORRELATIONS")
print("=" * 80)

top_correlations = target_correlation.head(15)

plt.figure(figsize=(10, 7))

top_correlations.sort_values().plot(
    kind="barh"
)

plt.title("Top Features Correlated With Fraud")
plt.xlabel("Correlation With Class")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 20: COMPARE FEATURE MEANS
# DIAGNOSTIC ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 20: FEATURE MEAN COMPARISON")
print("=" * 80)

feature_means = (
    df.groupby("Class")
    .mean(numeric_only=True)
    .T
)

feature_means.columns = [
    "Genuine",
    "Fraud"
]

display(feature_means)


# ==============================================================================
# STEP 21: FRAUD RATE BY TRANSACTION AMOUNT GROUP
# DIAGNOSTIC ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 21: FRAUD RATE BY AMOUNT GROUP")
print("=" * 80)

df["AmountGroup"] = pd.qcut(
    df["Amount"],
    q=10,
    duplicates="drop"
)

amount_fraud_rate = (
    df.groupby(
        "AmountGroup",
        observed=True
    )["Class"]
    .mean()
    * 100
)

display(
    amount_fraud_rate.to_frame(
        "Fraud Rate (%)"
    )
)


# ==============================================================================
# STEP 22: FRAUD RATE BY TIME GROUP
# DIAGNOSTIC ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 22: FRAUD RATE BY TIME GROUP")
print("=" * 80)

df["TimeGroup"] = pd.qcut(
    df["Time"],
    q=10,
    duplicates="drop"
)

time_fraud_rate = (
    df.groupby(
        "TimeGroup",
        observed=True
    )["Class"]
    .mean()
    * 100
)

display(
    time_fraud_rate.to_frame(
        "Fraud Rate (%)"
    )
)


# ==============================================================================
# STEP 23: DIAGNOSTIC SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 23: DIAGNOSTIC ANALYTICS SUMMARY")
print("=" * 80)

print("\nTotal Transactions:", len(df))

print(
    "Genuine Transactions:",
    (df["Class"] == 0).sum()
)

print(
    "Fraudulent Transactions:",
    (df["Class"] == 1).sum()
)

print(
    f"Fraud Rate: "
    f"{df['Class'].mean() * 100:.4f}%"
)

print(
    f"\nAverage Genuine Transaction Amount: "
    f"{df[df['Class'] == 0]['Amount'].mean():.2f}"
)

print(
    f"Average Fraud Transaction Amount: "
    f"{df[df['Class'] == 1]['Amount'].mean():.2f}"
)


# ==============================================================================
# STEP 24: PREPARE DATA FOR MACHINE LEARNING
# PREDICTIVE ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 24: PREPARE MACHINE LEARNING DATA")
print("=" * 80)

# Remove temporary analytical columns
df_model = df.drop(
    columns=[
        "AmountGroup",
        "TimeGroup"
    ],
    errors="ignore"
).copy()

# Separate features and target
X = df_model.drop(
    "Class",
    axis=1
)

y = df_model["Class"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeatures:")
print(X.columns.tolist())


# ==============================================================================
# STEP 25: TRAIN TEST SPLIT
# PREDICTIVE ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 25: TRAIN TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())


# ==============================================================================
# STEP 26: FEATURE SCALING
# PREDICTIVE ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 26: FEATURE SCALING")
print("=" * 80)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print("Feature scaling completed.")


# ==============================================================================
# STEP 27: HANDLE CLASS IMBALANCE USING SMOTE
# PREDICTIVE ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 27: HANDLE CLASS IMBALANCE USING SMOTE")
print("=" * 80)

print("\nBefore SMOTE:")

print(
    y_train.value_counts()
)

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train_scaled,
        y_train
    )
)

print("\nAfter SMOTE:")

print(
    pd.Series(
        y_train_smote
    ).value_counts()
)


# ==============================================================================
# STEP 28: LOGISTIC REGRESSION MODEL
# PREDICTIVE ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 28: LOGISTIC REGRESSION")
print("=" * 80)

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(
    X_train_smote,
    y_train_smote
)

print("Logistic Regression model trained successfully.")


# ==============================================================================
# STEP 29: LOGISTIC REGRESSION PREDICTION
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 29: LOGISTIC REGRESSION PREDICTION")
print("=" * 80)

y_pred_lr = logistic_model.predict(
    X_test_scaled
)

y_prob_lr = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]

print("Predictions generated successfully.")


# ==============================================================================
# STEP 30: LOGISTIC REGRESSION EVALUATION
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 30: LOGISTIC REGRESSION EVALUATION")
print("=" * 80)

lr_accuracy = accuracy_score(
    y_test,
    y_pred_lr
)

lr_precision = precision_score(
    y_test,
    y_pred_lr,
    zero_division=0
)

lr_recall = recall_score(
    y_test,
    y_pred_lr,
    zero_division=0
)

lr_f1 = f1_score(
    y_test,
    y_pred_lr,
    zero_division=0
)

lr_roc_auc = roc_auc_score(
    y_test,
    y_prob_lr
)

lr_pr_auc = average_precision_score(
    y_test,
    y_prob_lr
)

print(f"Accuracy:  {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall:    {lr_recall:.4f}")
print(f"F1 Score:  {lr_f1:.4f}")
print(f"ROC-AUC:   {lr_roc_auc:.4f}")
print(f"PR-AUC:    {lr_pr_auc:.4f}")


# ==============================================================================
# STEP 31: LOGISTIC REGRESSION CLASSIFICATION REPORT
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 31: LOGISTIC REGRESSION CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        y_pred_lr,
        target_names=[
            "Genuine",
            "Fraud"
        ],
        zero_division=0
    )
)


# ==============================================================================
# STEP 32: RANDOM FOREST MODEL
# PREDICTIVE ANALYTICS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 32: RANDOM FOREST")
print("=" * 80)

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

print("Random Forest model trained successfully.")


# ==============================================================================
# STEP 33: RANDOM FOREST PREDICTION
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 33: RANDOM FOREST PREDICTION")
print("=" * 80)

y_pred_rf = rf_model.predict(
    X_test_scaled
)

y_prob_rf = rf_model.predict_proba(
    X_test_scaled
)[:, 1]

print("Predictions generated successfully.")


# ==============================================================================
# STEP 34: RANDOM FOREST EVALUATION
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 34: RANDOM FOREST EVALUATION")
print("=" * 80)

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

rf_precision = precision_score(
    y_test,
    y_pred_rf,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    y_pred_rf,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    y_pred_rf,
    zero_division=0
)

rf_roc_auc = roc_auc_score(
    y_test,
    y_prob_rf
)

rf_pr_auc = average_precision_score(
    y_test,
    y_prob_rf
)

print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1 Score:  {rf_f1:.4f}")
print(f"ROC-AUC:   {rf_roc_auc:.4f}")
print(f"PR-AUC:    {rf_pr_auc:.4f}")


# ==============================================================================
# STEP 35: RANDOM FOREST CLASSIFICATION REPORT
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 35: RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        y_pred_rf,
        target_names=[
            "Genuine",
            "Fraud"
        ],
        zero_division=0
    )
)


# ==============================================================================
# STEP 36: LOGISTIC REGRESSION CONFUSION MATRIX
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 36: LOGISTIC REGRESSION CONFUSION MATRIX")
print("=" * 80)

cm_lr = confusion_matrix(
    y_test,
    y_pred_lr
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Genuine",
        "Fraud"
    ],
    yticklabels=[
        "Genuine",
        "Fraud"
    ]
)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 37: RANDOM FOREST CONFUSION MATRIX
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 37: RANDOM FOREST CONFUSION MATRIX")
print("=" * 80)

cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Genuine",
        "Fraud"
    ],
    yticklabels=[
        "Genuine",
        "Fraud"
    ]
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 38: MODEL COMPARISON
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 38: MODEL COMPARISON")
print("=" * 80)

results = pd.DataFrame({

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
        lr_roc_auc,
        rf_roc_auc
    ],

    "PR-AUC": [
        lr_pr_auc,
        rf_pr_auc
    ]
})

display(
    results.round(4)
)


# ==============================================================================
# STEP 39: MODEL COMPARISON VISUALIZATION
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 39: MODEL COMPARISON VISUALIZATION")
print("=" * 80)

metrics_to_plot = [
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
    "PR-AUC"
]

results_plot = (
    results
    .set_index("Model")[
        metrics_to_plot
    ]
)

results_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Machine Learning Model Performance Comparison"
)

plt.xlabel("Model")
plt.ylabel("Score")

plt.xticks(
    rotation=0
)

plt.ylim(
    0,
    1
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 40: ROC CURVE
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 40: ROC CURVE")
print("=" * 80)

fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    y_prob_lr
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    y_prob_rf
)

plt.figure(figsize=(10, 6))

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic Regression (AUC = {lr_roc_auc:.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {rf_roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title(
    "ROC Curve - Fraud Detection"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 41: PRECISION-RECALL CURVE
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 41: PRECISION-RECALL CURVE")
print("=" * 80)

precision_lr_curve, recall_lr_curve, _ = (
    precision_recall_curve(
        y_test,
        y_prob_lr
    )
)

precision_rf_curve, recall_rf_curve, _ = (
    precision_recall_curve(
        y_test,
        y_prob_rf
    )
)

plt.figure(figsize=(10, 6))

plt.plot(
    recall_lr_curve,
    precision_lr_curve,
    label=f"Logistic Regression (PR-AUC = {lr_pr_auc:.3f})"
)

plt.plot(
    recall_rf_curve,
    precision_rf_curve,
    label=f"Random Forest (PR-AUC = {rf_pr_auc:.3f})"
)

plt.title(
    "Precision-Recall Curve - Fraud Detection"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 42: RANDOM FOREST FEATURE IMPORTANCE
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 42: RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf_model.feature_importances_

})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\nTop 15 Important Features:")

display(
    feature_importance.head(15)
)


# ==============================================================================
# STEP 43: FEATURE IMPORTANCE VISUALIZATION
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 43: FEATURE IMPORTANCE VISUALIZATION")
print("=" * 80)

top_features = (
    feature_importance
    .head(15)
    .sort_values(
        "Importance"
    )
)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Features Used by Random Forest"
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 44: FRAUD RISK SCORE
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 44: FRAUD RISK SCORING")
print("=" * 80)

# Create prediction dataframe
prediction_results = X_test.copy()

prediction_results["Actual_Class"] = (
    y_test.values
)

prediction_results["Fraud_Probability"] = (
    y_prob_rf
)

prediction_results["Predicted_Class"] = (
    y_pred_rf
)


# ------------------------------------------------------------------------------
# Risk levels
# ------------------------------------------------------------------------------

prediction_results["Risk_Level"] = np.select(

    [
        prediction_results["Fraud_Probability"] >= 0.80,

        prediction_results["Fraud_Probability"] >= 0.50,

        prediction_results["Fraud_Probability"] < 0.50
    ],

    [
        "Very High Risk",
        "High Risk",
        "Low Risk"
    ],

    default="Low Risk"
)


print("\nSample Fraud Risk Predictions:")

display(
    prediction_results[
        [
            "Amount",
            "Actual_Class",
            "Fraud_Probability",
            "Predicted_Class",
            "Risk_Level"
        ]
    ].head(20)
)


# ==============================================================================
# STEP 45: RISK LEVEL DISTRIBUTION
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 45: RISK LEVEL DISTRIBUTION")
print("=" * 80)

risk_distribution = (
    prediction_results["Risk_Level"]
    .value_counts()
)

print("\nRisk Level Distribution:")

display(
    risk_distribution.to_frame(
        "Transaction Count"
    )
)

plt.figure(figsize=(9, 6))

sns.countplot(
    data=prediction_results,
    x="Risk_Level",
    order=[
        "Low Risk",
        "High Risk",
        "Very High Risk"
    ]
)

plt.title(
    "Transaction Risk Level Distribution"
)

plt.xlabel(
    "Risk Level"
)

plt.ylabel(
    "Number of Transactions"
)

plt.tight_layout()
plt.show()


# ==============================================================================
# STEP 46: SAVE PREDICTIONS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 46: SAVE FRAUD PREDICTIONS")
print("=" * 80)

output_file = (
    "credit_card_fraud_predictions.csv"
)

prediction_results.to_csv(
    output_file,
    index=False
)

print(
    f"Prediction results saved to: {output_file}"
)


# ==============================================================================
# STEP 47: FINAL MODEL SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 47: FINAL MODEL SUMMARY")
print("=" * 80)

print("\nMODEL PERFORMANCE")
print("-" * 80)

display(
    results.round(4)
)


# ==============================================================================
# STEP 48: BEST MODEL BASED ON PR-AUC
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 48: BEST MODEL SELECTION")
print("=" * 80)

best_model_index = (
    results["PR-AUC"].idxmax()
)

best_model = (
    results.loc[
        best_model_index,
        "Model"
    ]
)

best_pr_auc = (
    results.loc[
        best_model_index,
        "PR-AUC"
    ]
)

print(
    f"Best Model Based on PR-AUC: {best_model}"
)

print(
    f"PR-AUC: {best_pr_auc:.4f}"
)


# ==============================================================================
# STEP 49: BUSINESS INSIGHTS
# ==============================================================================

print("\n" + "=" * 80)
print("STEP 49: FINAL BUSINESS INSIGHTS")
print("=" * 80)

print("""
CREDIT CARD FRAUD DETECTION - BUSINESS INSIGHTS

1. FRAUD IS AN EXTREMELY RARE EVENT
   The dataset contains a very small percentage of fraudulent
   transactions compared with genuine transactions.

2. ACCURACY ALONE IS NOT SUFFICIENT
   A model can achieve high accuracy simply by predicting most
   transactions as genuine. Therefore, precision, recall, F1,
   ROC-AUC and especially PR-AUC should be considered.

3. RECALL IS IMPORTANT
   Higher recall means the system identifies more fraudulent
   transactions and reduces missed fraud.

4. PRECISION IS ALSO IMPORTANT
   Higher precision reduces the number of genuine customers
   incorrectly flagged as fraudulent.

5. F1-SCORE BALANCES PRECISION AND RECALL
   F1-score is useful when both false positives and false
   negatives matter.

6. PR-AUC IS PARTICULARLY IMPORTANT
   Precision-Recall AUC is useful for evaluating models when
   the positive class is extremely rare.

7. SMOTE ADDRESSES CLASS IMBALANCE
   Synthetic minority samples were generated only on the
   training data to help the models learn the fraud class.

8. RANDOM FOREST CAN CAPTURE NONLINEAR RELATIONSHIPS
   Random Forest can identify complex relationships between
   transaction characteristics and fraud.

9. FRAUD PROBABILITY SUPPORTS RISK-BASED DECISIONS
   Instead of making only a binary fraud/non-fraud decision,
   the model can provide a probability-based risk score.

10. BUSINESS THRESHOLDS SHOULD BE COST-BASED
    In a real banking environment, the probability threshold
    should be selected according to the financial cost of
    missed fraud versus the operational/customer cost of
    false alarms.

11. HIGH-RISK TRANSACTIONS CAN RECEIVE ADDITIONAL VERIFICATION
    Examples include OTP verification, temporary transaction
    holds, customer confirmation, or manual review.

12. MODEL MONITORING IS REQUIRED
    Fraud patterns can change over time, so a production
    system should be continuously monitored and retrained.
""")


# ==============================================================================
# STEP 50: PROJECT COMPLETION SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 80)

print("""
Analytics Covered
-----------------
Descriptive Analytics
Diagnostic Analytics
Predictive Analytics

Machine Learning Models
-----------------------
1. Logistic Regression
2. Random Forest

Class Imbalance Technique
-------------------------
SMOTE

Evaluation Metrics
------------------
Accuracy
Precision
Recall
F1 Score
ROC-AUC
PR-AUC

Visualizations
--------------
Fraud Distribution
Amount Distribution
Time Distribution
Box Plot
Correlation Matrix
Feature Correlation
Confusion Matrices
ROC Curve
Precision-Recall Curve
Feature Importance
Risk Distribution

Output
------
credit_card_fraud_predictions.csv
""")

print("=" * 80)
print("END OF CREDIT CARD FRAUD DETECTION PROJECT")
print("=" * 80)