# 📱 Customer Churn Telco Prediction

A machine learning-powered web application built with **Streamlit** that predicts whether a telecom customer is likely to churn, enabling companies to take proactive retention actions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Input Features](#input-features)
- [Model Pipeline](#model-pipeline)

---

## 🔍 Overview

The telecommunications industry is highly competitive, and customers can freely switch providers — a behavior known as **Customer Churn**. This application uses a supervised machine learning classification model to predict the likelihood of a customer churning, allowing companies to intervene with special offers or improved service before it's too late.

---

## ❗ Problem Statement

The rapid growth of the telecom industry has led to fierce competition among service providers. Customers can switch providers at will, which directly reduces company revenue. Identifying at-risk customers before they churn is critical for business sustainability.

> **Goal:** Build a machine learning model to classify whether a telecom customer will churn or not, based on historical customer data.

---

## ✅ Solution

Based on data analysis, two key strategies are recommended to retain customers:

1. **Targeted Offers** — Provide special treatment (e.g., exclusive packages or discounts) to customers predicted to churn by the ML model.
2. **Service Improvement** — Continuously improve the quality of services provided to customers.

---

## ✨ Features

- 🔮 Real-time churn prediction via an interactive web form
- 📊 Input validation and preprocessing (winsorization, log transformation, encoding, scaling, PCA)
- 🌐 Bilingual support — **English** and **Bahasa Indonesia**
- ⚠️ Clear output: probability score + churn/no-churn label
- 📄 Multi-page Streamlit app (Home, Problem, Solution, Contact)

---

## 📁 Project Structure

```
├── 1_Home.py              # Main prediction page with user input form
├── pages/
│   ├── 2_Problem.py       # Problem statement (EN & ID)
│   ├── 3_Solution.py      # Proposed solutions (EN & ID)
│   └── 4_Contact.py       # Contact information
├── datasets/
│   └── train.csv          # Training dataset (used for preprocessing reference)
├── model_clf.pkl          # Trained Random Forest classifier
├── scaler_clf.pkl         # Fitted StandardScaler
├── pca_clf.pkl            # Fitted PCA transformer
└── README.md
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scipy` | Winsorization (outlier handling) |
| `scikit-learn` | Scaler, PCA, Random Forest model |
| `pickle` | Model serialization / loading |

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/customer-churn-telco.git
   cd customer-churn-telco
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install streamlit pandas numpy scipy scikit-learn
   ```

4. **Ensure model artifacts and dataset are in place**
   - `model_clf.pkl`
   - `scaler_clf.pkl`
   - `pca_clf.pkl`
   - `datasets/train.csv`

---

## 🚀 Usage

```bash
streamlit run pages/1_Home.py
```

Then open your browser at `http://localhost:8501`.

---

## 📥 Input Features

| Feature | Description |
|---|---|
| `state` | Customer's US state |
| `account_length` | Duration as a customer (months) |
| `area_code` | Customer's area code |
| `international_plan` | Has international plan (yes/no) |
| `voice_mail_plan` | Has voicemail plan (yes/no) |
| `number_vmail_messages` | Total voicemail messages |
| `total_day_minutes` | Total daytime call minutes |
| `total_day_calls` | Total daytime calls |
| `total_day_charge` | Total daytime charge |
| `total_eve_minutes` | Total evening call minutes |
| `total_eve_calls` | Total evening calls |
| `total_eve_charge` | Total evening charge |
| `total_night_minutes` | Total night call minutes |
| `total_night_calls` | Total night calls |
| `total_night_charge` | Total night charge |
| `total_intl_minutes` | Total international minutes |
| `total_intl_calls` | Total international calls |
| `total_intl_charge` | Total international charge |
| `number_customer_service_calls` | Number of customer service calls made |

---

## 🧪 Model Pipeline

The prediction pipeline follows these steps:

1. **Outlier Handling** — Winsorization (1st–99th percentile) for normally distributed features
2. **Log Transformation** — Applied to skewed/bimodal features (`number_vmail_messages`, `total_intl_calls`, `number_customer_service_calls`)
3. **Label Encoding** — Binary encoding for `international_plan` and `voice_mail_plan`
4. **One-Hot Encoding** — Applied to `state` and `area_code`
5. **Feature Scaling** — StandardScaler loaded from `scaler_clf.pkl`
6. **Dimensionality Reduction** — PCA loaded from `pca_clf.pkl`
7. **Prediction** — Random Forest Classifier loaded from `model_clf.pkl`
