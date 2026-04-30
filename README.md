# Customer Churn Prediction & Retention Strategy Engine

A complete end-to-end machine learning project to predict customer churn 
in a telecom company and generate actionable retention strategies.

🔗 **Live App:** [Customer Churn Predictor](https://customer-churn-prediction-kdduvlj7dtjukbrapplpftn.streamlit.app)

---

## Project Overview

Customer churn is a critical business problem — losing customers means losing revenue. 
This project goes beyond just building a model. It identifies *which* customers will 
leave, *why* they leave, and *what the business should do* about each segment.

---

## What This Project Does

- Analyzes 7,000+ telecom customers to find churn patterns
- Compares 5 ML models and selects the best using multiple metrics
- Engineers new features to improve model performance
- Tunes the classification threshold to maximize recall
- Segments customers by risk and business value
- Quantifies revenue at risk ($2.3M identified)
- Provides specific retention actions per segment
- Deploys as a live interactive web application

---

## Key Results

| Metric | Score |
|--------|-------|
| Model | XGBoost (GridSearchCV tuned) |
| Recall | 79.9% |
| ROC-AUC | 85.2% |
| Threshold | 0.55 (tuned for recall) |
| Customers at risk | 568 predicted churners |
| Revenue at risk | $2.3M total CLTV |
| Priority segment | 198 customers — $1M+ CLTV |

---

## Business Insights from EDA

- Month-to-month customers churn at **42.7%** vs **2.8%** for two-year contracts
- **33% of churners** left for a competitor — pricing/offers are the main driver
- Fiber optic users churn significantly more than DSL users
- New customers (0-12 months) churn at **47.4%** — nearly half leave in year one
- Electronic check payment users show highest churn rate

---

## Customer Segments & Retention Strategy

| Segment | Customers | Avg CLTV | Action |
|---------|-----------|----------|--------|
| Priority — Retain Now | 198 | $5,284 | Personal outreach + premium discount |
| High Risk — Monitor | 178 | $3,786 | Automated retention email |
| High Risk — Low Value | 130 | $2,491 | Low-cost chatbot offer |
| Medium Risk — Engage | 304 | $4,425 | Loyalty program enrollment |
| Stable — Maintain | 599 | $4,657 | Regular engagement + upsell |

---

## Project Structure

| File | Description |
|------|-------------|
| `churn_project2.ipynb` | Main notebook — EDA, modeling, SQL, segmentation |
| `app.py` | Streamlit web application |
| `churn_model.pkl` | Trained XGBoost model |
| `threshold.pkl` | Optimized classification threshold (0.55) |
| `scaler.pkl` | Fitted StandardScaler |
| `scale_cols.pkl` | Columns to scale |
| `Telco_customer_churn.csv` | Dataset (7,043 customers, 33 features) |
| `requirements.txt` | Python dependencies |

---

## Notebook Contents

**Part 1** — Setup & Data Loading  
**Part 2** — EDA & Business Insights (Churn Reason, CLTV, patterns)  
**Part 3** — Preprocessing & Feature Engineering  
**Part 4** — Model Comparison (LR, DT, RF, Gradient Boosting, XGBoost)  
**Part 5** — Hyperparameter Tuning & Threshold Optimization  
**Part 6** — Feature Importance & Customer Segmentation  
**Part 7** — SQL Analysis (6 queries: GROUP BY, CASE WHEN, Window Functions)  
**Part 8** — Streamlit App Deployment  

---

## Tech Stack

- **Python** — Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **SQL** — SQLite with window functions (RANK, PARTITION BY)
- **Deployment** — Streamlit Cloud
- **Version Control** — Git, GitHub

---

## How to Run Locally

```bash
git clone https://github.com/PrathamjitSinghWalia/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
streamlit run app.py
```
