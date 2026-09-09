# 💳 Credit Card Fraud Detection Using Logistic Regression

## 📌 Project Overview

Credit card fraud detection is a binary classification problem where the goal is to identify whether a financial transaction is **legitimate** or **fraudulent**.

This project uses **Logistic Regression** to predict fraudulent transactions while addressing the severe class imbalance present in the dataset.

The project follows a complete analytics workflow:

**Data Loading → Data Understanding → Data Cleaning → Descriptive Analysis → Diagnostic Analysis → Data Visualization → Correlation Analysis → Feature Preparation → Machine Learning → Model Evaluation → Fraud Probability Prediction → Fraud Risk Classification → Business Insights → Export Results**

---

## 📂 Dataset

The dataset used in this project is the **Credit Card Fraud Detection Dataset** from Kaggle.

The original CSV file is approximately **150 MB**, so it is not included directly in this GitHub repository.

Instead, download the dataset from Kaggle:

[Credit Card Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?utm_source=chatgpt.com)

### Dataset Statistics

| Property | Description |
|---|---|
| Total Transactions | 284,807 |
| Fraudulent Transactions | 492 |
| Genuine Transactions | 284,315 |
| Fraud Rate | Approximately 0.172% |
| Features | 30 |
| Target Variable | `Class` |
| Dataset Type | Binary Classification |

### Dataset Features

The dataset contains the following columns:

- `Time` — Seconds elapsed between each transaction and the first transaction
- `V1`–`V28` — PCA-transformed/anonymized numerical features
- `Amount` — Transaction amount
- `Class` — Target variable

### Target Variable

| Class | Meaning |
|---:|---|
| `0` | Genuine transaction |
| `1` | Fraudulent transaction |

> ⚠️ **Important:** The dataset is highly imbalanced. Fraudulent transactions represent only a very small percentage of all transactions. Therefore, accuracy alone is not sufficient for evaluating the model.

---

## 📥 How to Get the Dataset

1. Open the Kaggle dataset:

   [Download Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?utm_source=chatgpt.com)

2. Download the dataset from Kaggle.

3. Extract the downloaded ZIP file.

4. Place `creditcard.csv` in the same directory as the Jupyter Notebook.

The expected project structure is:

```text
Credit-Card-Fraud-Detection/
│
├── credit_card_fraud_detection.ipynb
├── README.md
├── creditcard.csv
│
├── credit_card_fraud_predictions.csv
├── credit_card_fraud_model_evaluation.csv
└── credit_card_fraud_risk_summary.csv
```

> The `creditcard.csv` file is **not required to be uploaded to GitHub**. Users can download it directly from Kaggle and place it locally before running the notebook.

---

## 🔬 Project Stages

### 1. Data Loading

The dataset is loaded using Pandas:

```python
df = pd.read_csv("creditcard.csv")
```

The CSV is kept locally because of its large file size.

---

### 2. Data Understanding

The dataset structure is examined using:

```python
df.shape
df.head()
df.info()
df.dtypes
df["Class"].value_counts()
```

This helps understand the number of records, columns, data types, and target distribution.

---

### 3. Data Cleaning

The dataset is checked for:

- Missing values
- Duplicate records
- Incorrect data types
- Unnecessary columns

Duplicate records are removed when present.

---

### 4. Descriptive Analysis

Statistical summaries are generated using:

```python
df.describe().T
```

The analysis includes:

- Transaction amount statistics
- Genuine transaction statistics
- Fraud transaction statistics
- Fraud percentage
- Minimum and maximum transaction amounts

Fraud rate is calculated as:

\[
Fraud\ Rate =
\frac{Number\ of\ Fraudulent\ Transactions}
{Total\ Transactions}
\times 100
\]

---

### 5. Diagnostic Analysis

Diagnostic analysis investigates potential differences between genuine and fraudulent transactions.

The project analyzes:

- Transaction amount
- Transaction time
- Fraud rate by amount range
- Fraud rate by time period
- Feature-level differences between fraud and genuine transactions

---

### 6. Data Visualization

Visualizations are created to understand the dataset and fraud patterns.

Examples include:

- Genuine vs Fraud transaction counts
- Transaction amount distribution
- Fraud transaction amount distribution
- Transaction time distribution
- Fraud rate by transaction amount
- Fraud rate by time period

---

### 7. Correlation Analysis

A correlation matrix is created to investigate relationships between numerical variables.

Special attention is given to correlations with the target variable:

```python
correlation_matrix["Class"].sort_values(ascending=False)
```

A heatmap is also generated to visualize feature relationships.

> Because `V1`–`V28` are anonymized/PCA-derived variables, their individual business meanings are not assumed.

---

### 8. Feature Preparation

The target variable is separated from the input features:

```python
X = df.drop(columns=["Class"])
y = df["Class"]
```

The dataset is divided into training and testing sets using stratification:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Numerical features are standardized using `StandardScaler`.

---

### 9. Machine Learning

The project uses **Logistic Regression** as the machine-learning algorithm.

```python
LogisticRegression(
    max_iter=2000,
    random_state=42
)
```

Because the dataset contains extreme class imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied only to the training data.

This helps the model learn patterns associated with the minority fraud class.

---

### 10. Model Evaluation

The model is evaluated using multiple metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

### Accuracy

\[
Accuracy =
\frac{TP+TN}{TP+TN+FP+FN}
\]

### Precision

\[
Precision =
\frac{TP}{TP+FP}
\]

Precision measures how many transactions predicted as fraud were actually fraudulent.

### Recall

\[
Recall =
\frac{TP}{TP+FN}
\]

Recall is particularly important in fraud detection because it measures how many actual fraudulent transactions were detected.

### F1-Score

\[
F1 =
2\times
\frac{Precision\times Recall}
{Precision+Recall}
\]

### PR-AUC

Precision-Recall AUC is especially useful for highly imbalanced classification problems.

---

### 11. Fraud Probability Prediction

Instead of only producing a binary fraud prediction, the Logistic Regression model generates a fraud probability:

```python
fraud_probability = model.predict_proba(X_test_scaled)[:, 1]
```

Example:

```text
Transaction → Fraud Probability
Transaction A → 0.92
Transaction B → 0.74
Transaction C → 0.18
```

This allows transactions to be prioritized according to their estimated fraud probability.

---

### 12. Fraud Risk Classification

Fraud probabilities are converted into risk categories.

| Fraud Probability | Risk Level |
|---:|---|
| `< 0.50` | Low Risk |
| `0.50 – < 0.80` | High Risk |
| `>= 0.80` | Very High Risk |

> These thresholds are illustrative for this project and should be optimized using business costs, model calibration, and operational requirements in a production fraud-detection system.

---

### 13. Business Insights

The analysis provides insights such as:

- Fraud represents a very small proportion of all transactions.
- Accuracy can be misleading because of extreme class imbalance.
- Recall is important because missed fraud can result in financial losses.
- Precision is important because excessive false alarms can inconvenience legitimate customers.
- Fraud probability can be used to prioritize transactions for investigation.
- High-risk and very-high-risk transactions can receive additional review.
- Fraud detection systems should continuously monitor changing transaction patterns.

### Business Trade-off

There are two major types of errors:

**False Negative**

A fraudulent transaction is classified as genuine.

Potential consequence:

> Financial loss.

**False Positive**

A genuine transaction is classified as fraudulent.

Potential consequence:

> Customer inconvenience and unnecessary investigation.

Therefore, an effective fraud detection system must balance **fraud detection** with **customer experience**.

---

### 14. Export Results

The project exports the following files:

```text
credit_card_fraud_predictions.csv
credit_card_fraud_model_evaluation.csv
credit_card_fraud_risk_summary.csv
```

These files contain:

- Actual transaction class
- Predicted transaction class
- Fraud probability
- Risk level
- Model evaluation metrics
- Fraud risk summary

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook
- Kaggle Dataset

---

## 📦 Installation

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

---

## ▶️ How to Run

### Step 1 — Download the Dataset

Download `creditcard.csv` from:

[Kaggle — Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?utm_source=chatgpt.com)

### Step 2 — Place the Dataset

Place:

```text
creditcard.csv
```

in the same directory as the Jupyter Notebook.

### Step 3 — Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 4 — Run the Notebook

Open:

```text
credit_card_fraud_detection.ipynb
```

and execute the cells sequentially.

---

## 📊 Expected Outputs

The notebook produces:

### Data Analysis

- Dataset information
- Statistical summaries
- Fraud distribution
- Fraud rate analysis
- Correlation analysis

### Visualizations

- Class distribution
- Transaction amount distributions
- Time distributions
- Fraud-rate analysis
- Correlation heatmaps
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Risk distribution

### Machine Learning

- Logistic Regression model
- SMOTE-balanced training data
- Fraud predictions
- Fraud probabilities
- Risk classifications

### Exported Files

```text
credit_card_fraud_predictions.csv
credit_card_fraud_model_evaluation.csv
credit_card_fraud_risk_summary.csv
```

---

## ⚠️ Limitations

1. The dataset contains anonymized PCA-transformed features.
2. Individual `V1`–`V28` features cannot be interpreted as specific business attributes.
3. The dataset represents transactions from a limited historical period.
4. The model is intended as an educational/project implementation rather than a production banking system.
5. Risk thresholds used in this project are illustrative.
6. Production fraud systems would require continuous monitoring and retraining.
7. Probability calibration and cost-sensitive threshold optimization could further improve the system.

---

## 🚀 Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Probability calibration
- Cost-sensitive learning
- Threshold optimization
- Cross-validation
- Model monitoring
- Real-time fraud scoring
- Feature engineering
- Explainable AI
- Advanced anomaly detection
- Ensemble models

---

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

- Binary classification
- Highly imbalanced datasets
- Data cleaning
- Descriptive analytics
- Diagnostic analytics
- Data visualization
- Correlation analysis
- Feature scaling
- SMOTE
- Logistic Regression
- Classification metrics
- ROC-AUC
- PR-AUC
- Confusion matrices
- Fraud probability prediction
- Risk classification
- Business-oriented machine-learning interpretation
- Exporting analytical results

---

## 🏁 Conclusion

This Credit Card Fraud Detection project demonstrates how machine learning can be applied to identify potentially fraudulent financial transactions.

The project uses **Logistic Regression** combined with **SMOTE** to address the extreme class imbalance in the dataset.

Rather than relying only on accuracy, the project evaluates the model using **Precision, Recall, F1-score, ROC-AUC, and PR-AUC**, providing a more meaningful assessment of fraud-detection performance.

The final system also converts fraud probabilities into **Low Risk, High Risk, and Very High Risk** categories, making the model outputs easier to interpret from a business perspective.

The large dataset is intentionally not stored in this GitHub repository. Users can download the original `creditcard.csv` file directly from Kaggle using the link provided above.

---

## 👨‍💻 Author

**Avinassh Kotta**

Machine Learning | Data Analytics | Python | NLP