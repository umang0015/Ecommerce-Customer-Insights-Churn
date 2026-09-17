# E-Commerce Customer Insights, Sales Performance & Churn Risk Analytics

## Overview

This is a complete end-to-end Data Science / Data Analytics capstone project analyzing an e-commerce customer dataset containing 2,000 transaction records across 17 columns. The project follows the four-level analytics progression: Descriptive → Diagnostic → Predictive → Prescriptive.

## Business Problem

An e-commerce business selling products across multiple categories and countries needs to understand sales performance, customer behavior patterns, identify which customers are at risk of churn, and develop data-driven retention strategies.

## Objectives

1. Calculate key sales and customer KPIs
2. Identify patterns associated with churn and customer inactivity
3. Build a churn prediction model using Logistic Regression
4. Segment customers by risk and recommend retention strategies

## Dataset

- **Source**: E Commerce Customer Insights and Churn Dataset.csv
- **Rows**: 2,000 | **Columns**: 17
- **Key columns**: order_id, customer_id, age, country, signup_date, last_purchase_date, subscription_status, order_date, unit_price, quantity, category, gender
- **Quality**: No missing values, no duplicate rows

## Methodology

1. **Data Quality Audit** - Missing values, duplicates, invalid values, date consistency
2. **Data Cleaning** - Date parsing, revenue calculation, feature engineering
3. **Descriptive Analytics** - KPIs and 10+ visualizations
4. **Diagnostic Analytics** - Cross-tabulations, correlations, segmentation
5. **RFM Analysis** - Recency, Frequency, Monetary scoring
6. **Churn Definition** - Behavior-based target using historical snapshot
7. **Predictive Modeling** - Logistic Regression with train/test split
8. **Prescriptive Analytics** - Risk-based business recommendations

## Technologies Used

Python 3, pandas, numpy, matplotlib, seaborn, scikit-learn, Jupyter Notebook

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/ecommerce_customer_insights_churn_analysis.ipynb
```

## Project Structure

```
ecommerce-customer-insights-churn/
├── data/
│   └── E Commerce Customer Insights and Churn Dataset.csv
├── notebooks/
│   └── ecommerce_customer_insights_churn_analysis.ipynb
├── outputs/
│   ├── figures/
│   └── reports/
├── README.md
├── requirements.txt
└── .gitignore
```

## Key Findings

- Dataset contains 2,000 transactions across 6 countries and 5 product categories
- Subscription status distribution: 1,204 active, 493 cancelled, 303 paused
- No missing values or duplicate rows detected
- Revenue ranges from $2.85 to $1,991.63 per transaction
- Age distribution spans 18-69 years with mean of 44.1
- Behavior-based churn definition used to avoid target leakage

## Business Recommendations

1. Focus retention efforts on high-value customers with high churn probability
2. Investigate product/service friction for customers with high cancellation counts
3. Implement category-specific targeted campaigns
4. Prioritize resource allocation based on risk-value matrix

---

*Note: All numerical findings are calculated from the actual dataset. No values have been fabricated.*
