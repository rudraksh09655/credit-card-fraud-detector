# 🔒 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.0-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> An end-to-end Machine Learning project to detect fraudulent credit card transactions using Random Forest Classifier with SMOTE for class imbalance, deployed as an interactive web app on Streamlit Cloud.

🚀 **Live Demo:** [credit-card-fraud-detector.streamlit.app](https://credit-card-fraud-detector-bbiumrjgheqhqb39jdpavt.streamlit.app)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [ML Pipeline](#ml-pipeline)
- [Key Techniques](#key-techniques)
- [Model Results](#model-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Web App](#web-app)
- [Tech Stack](#tech-stack)
- [Interview Q&A](#interview-qa)

---

## 🔍 Overview

Credit card fraud is a major financial problem worldwide. This project builds a complete machine learning pipeline that:

- Trains on **284,807 real-world transactions** from European cardholders
- Detects fraud with **ROC-AUC of 0.999**
- Handles extreme class imbalance using **SMOTE**
- Deploys as a **live interactive web app** where users can input transaction details and get real-time fraud predictions

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total Transactions | 284,807 |
| Fraud Cases | 492 (0.17%) |
| Legitimate Cases | 284,315 (99.83%) |
| Features | 30 (V1–V28, Amount, Time) |
| Target | Class (0 = Legitimate, 1 = Fraud) |

### Feature Description

| Feature | Description |
|---|---|
| V1 – V28 | PCA-transformed features (anonymized by the bank to protect privacy) |
| Amount | Transaction amount in dollars |
| Time | Seconds elapsed between this transaction and the first transaction |
| Class | Target variable — 0 (Legitimate) or 1 (Fraud) |

> **Note:** The original features were anonymized using PCA (Principal Component Analysis) by the bank for confidentiality reasons. Only `Amount` and `Time` are the original features.

---

## ❓ Problem Statement

This is a **binary classification problem** with a major challenge:

```
Normal Transactions  → 284,315  (99.83%)
Fraud Transactions   →     492  (0.17%)
```

A naive model that predicts **everything as normal** would get **99.83% accuracy** — but would detect **zero fraud cases**. This is why accuracy is a misleading metric here and why handling class imbalance is critical.

---

## 🧠 ML Pipeline

```
Raw Data (creditcard.csv)
        │
        ▼
 Exploratory Data Analysis
 - Class distribution
 - Transaction amount distribution
 - Feature correlation with fraud
        │
        ▼
    Preprocessing
 - Scale Amount using StandardScaler
 - Scale Time using StandardScaler
 - Drop original Amount and Time columns
        │
        ▼
   Train-Test Split (80/20)
 - stratify=y to maintain class ratio
 - Split BEFORE applying SMOTE
        │
        ▼
  SMOTE on Training Data Only
 - Oversample minority (fraud) class
 - Balance: 227,451 normal vs 227,451 fraud
        │
        ▼
    Model Training
 - Logistic Regression (baseline)
 - Random Forest (final model)
        │
        ▼
    Evaluation
 - Confusion Matrix
 - Classification Report
 - ROC-AUC Score
 - ROC Curve
        │
        ▼
  Save Model (joblib)
 - fraud_model.pkl
 - scaler.pkl
        │
        ▼
  Streamlit Web App
 - Load saved model
 - Accept user input
 - Return prediction + probability
```

---

## 🔑 Key Techniques

### 1. SMOTE (Synthetic Minority Oversampling Technique)
Instead of simply duplicating fraud cases, SMOTE **creates synthetic fraud samples** by interpolating between existing fraud transactions. This gives the model more diverse fraud patterns to learn from.

```python
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```

> ⚠️ **Critical:** SMOTE is applied **only on training data**, not on test data. Applying it before the split would cause **data leakage** — synthetic test samples contaminating training, inflating performance unrealistically.

### 2. Why ROC-AUC over Accuracy?
- **Accuracy** is misleading with imbalanced data (99.83% accuracy by predicting all normal)
- **ROC-AUC** measures how well the model distinguishes between fraud and normal across all thresholds
- **Precision** = Of all predicted frauds, how many were actually fraud?
- **Recall** = Of all actual frauds, how many did we catch?

### 3. Random Forest over Logistic Regression
- Handles non-linear relationships in PCA features
- Robust to outliers (fraud transactions often have unusual amounts)
- Provides feature importance scores
- Significantly outperforms Logistic Regression baseline on this dataset

---

## 📈 Model Results

| Model | ROC-AUC | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|---|---|---|---|---|
| Logistic Regression | ~0.97 | ~0.88 | ~0.91 | ~0.89 |
| **Random Forest** | **~0.999** | **~0.97** | **~0.82** | **~0.89** |

### Confusion Matrix (Random Forest)
```
                 Predicted Normal    Predicted Fraud
Actual Normal        56,854               10
Actual Fraud             17               81
```
- ✅ 56,854 legitimate transactions correctly identified
- ✅ 81 fraud transactions correctly caught
- ⚠️ Only 17 fraud cases missed (False Negatives)
- ⚠️ Only 10 false alarms (False Positives)

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv          ← dataset (not uploaded to GitHub)
│
├── model/
│   ├── fraud_model.pkl         ← trained Random Forest model
│   └── scaler.pkl              ← fitted StandardScaler
│
├── notebook/
│   └── analysis.ipynb          ← EDA + model training notebook
│
├── app.py                      ← Streamlit web application
├── requirements.txt            ← Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/rudraksh09655/credit-card-fraud-detector.git
cd credit-card-fraud-detector
```

**2. Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download dataset**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

---

## ▶️ How to Run

### Run the Jupyter Notebook (Training)
```bash
cd notebook
jupyter notebook analysis.ipynb
```
Run all cells to train the model and save `fraud_model.pkl` and `scaler.pkl` to the `model/` folder.

### Run the Streamlit App (Web App)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 🌐 Web App

The Streamlit web app allows users to:

- Input **transaction amount** and **time**
- Adjust **V1–V28 PCA feature sliders**
- Click **"Analyze Transaction"** to get instant prediction
- View **fraud probability** and **legitimacy probability**

### App Preview

```
┌─────────────────────────────────────────────────────────┐
│  Sidebar               │  Main Panel                    │
│  ─────────             │  ─────────────────────────     │
│  About This Model      │  🔒 Credit Card Fraud          │
│  • Random Forest       │     Detection                  │
│  • 284,807 txns        │                                │
│  • SMOTE               │  Transaction Amount: [100.00]  │
│  • ROC-AUC: 0.999      │  Time: [50000.00]              │
│                        │                                │
│                        │  V1 ──●────────── 0.00         │
│                        │  V2 ──────●────── 0.00         │
│                        │  ...                           │
│                        │                                │
│                        │  [🔍 Analyze Transaction]      │
│                        │                                │
│                        │  ✅ LEGITIMATE — 96.4%         │
│                        │  Fraud Prob: 3.60%             │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| ML Library | Scikit-learn |
| Imbalance Handling | Imbalanced-learn (SMOTE) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Saving | Joblib |
| Notebook | Jupyter |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## 💡 Interview Q&A

**Q: Why not use accuracy as the evaluation metric?**
> With only 0.17% fraud cases, a model predicting everything as normal gets 99.83% accuracy but detects zero fraud. ROC-AUC and Precision-Recall are better metrics for imbalanced datasets.

**Q: Why did you apply SMOTE only on training data?**
> Applying SMOTE before the train-test split causes data leakage — synthetic samples in the test set would be derived from training data, inflating performance metrics unrealistically. The test set must always represent real-world unseen data.

**Q: Why Random Forest over Logistic Regression?**
> Random Forest handles non-linear relationships, is robust to outliers, provides feature importance, and significantly outperformed Logistic Regression (0.999 vs 0.97 ROC-AUC) on this dataset.

**Q: What are V1–V28 features?**
> They are PCA-transformed features from the original transaction data, anonymized by the bank to protect user privacy. PCA reduces dimensionality while preserving maximum variance in the data.

**Q: What is SMOTE?**
> Synthetic Minority Oversampling Technique creates new synthetic fraud samples by interpolating between existing fraud cases, rather than simply duplicating them. This gives the model more diverse fraud patterns to learn from.

---

## 👨‍💻 Author

**Rudraksh Gupta**
- GitHub: [@rudraksh09655](https://github.com/rudraksh09655)
- LinkedIn: [Rudraksh Gupta](https://www.linkedin.com/in/rudraksh-gupta-a275072a6/)
- Email: rudrakshgupta50@gmail.com

---

