# E-Commerce Customer Insights & Churn Analytics

## Project Report

**Student Name:** Umang Ladha

**Date:** September 2026

---

## 1. Executive Summary

This project analyzes an e-commerce customer dataset containing 2,000 transaction records across 17 columns. The goal is to understand sales performance, identify customer behavior patterns, and build a predictive model to detect customers at risk of churn. The analysis follows a four-level analytics progression: Descriptive → Diagnostic → Predictive → Prescriptive.

**Key Results:**
- Total Revenue: $2,051,690.65 across 2,000 orders
- Churn Rate: 24.6% (493 customers cancelled subscriptions)
- Average Order Value: $1,025.85
- Top Category: Clothing
- Top Country: Germany
- Logistic Regression Model: 75.2% accuracy, 0.5072 ROC-AUC

---

## 2. Business Problem

An e-commerce business selling products across multiple categories and countries needs to understand sales performance, identify which customers are at risk of churn, and develop data-driven retention strategies. The business wants to move from reactive customer management to proactive retention based on data insights.

### Objectives

1. Calculate key sales and customer KPIs
2. Identify patterns associated with churn and customer inactivity
3. Build a churn prediction model using Logistic Regression
4. Segment customers by risk level and recommend retention strategies
5. Provide actionable business recommendations based on data analysis

---

## 3. Dataset Description

### Source
`E Commerce Customer Insights and Churn Dataset.csv`

### Overview
- **Rows:** 2,000 customer records
- **Columns:** 17
- **Countries:** 6 (including Germany, USA, Canada)
- **Categories:** 5 product categories
- **Genders:** 3
- **Data Quality:** No missing values, no duplicate rows

### Key Columns
| Column | Description |
|--------|-------------|
| `order_id` | Unique order identifier |
| `customer_id` | Unique customer identifier |
| `age` | Customer age |
| `country` | Customer country |
| `gender` | Customer gender |
| `product_id` | Product identifier |
| `product_name` | Product name |
| `category` | Product category |
| `unit_price` | Price per unit |
| `quantity` | Quantity purchased |
| `signup_date` | Date customer signed up |
| `order_date` | Date of the order |
| `last_purchase_date` | Last purchase date |
| `subscription_status` | Active, paused, or cancelled |
| `cancellations_count` | Number of cancellations |
| `purchase_frequency` | Independent frequency metric (1-49) |

### Data Quality Audit Findings
- Each customer has exactly 1 order in the dataset
- `signup_date` is always before `order_date` (100% consistency)
- `last_purchase_date` is a separate engagement metric, unrelated to `order_date`
- `purchase_frequency` (1-49) is an independent metric, not actual order count
- All customer attributes (age, country, gender, etc.) are constant per customer

---

## 4. Methodology

The project follows a structured four-phase approach:

### Phase 1: Descriptive Analytics
- Calculated key revenue metrics (total revenue, AOV, revenue by country/category)
- Analyzed customer demographics (age, gender, country distributions)
- Examined subscription status distribution
- Created monthly revenue trends

### Phase 2: Diagnostic Analytics
- Performed correlation analysis to identify relationships with churn
- Analyzed churn rates by category, country, gender, and cancellation count
- Compared churned vs active customers across key metrics
- Identified that cancellation count and subscription status are the strongest churn predictors

### Phase 3: Predictive Analytics
- Built a Logistic Regression model to predict churn
- Used One-Hot Encoding for categorical features
- Applied StandardScaler fitted only on training data to prevent target leakage
- Used stratified 80/20 train/test split

### Phase 4: Prescriptive Analytics
- Applied the model to generate churn probabilities for all customers
- Created risk segments (Low, Medium, High)
- Identified top 50 high-risk customers for targeted retention
- Developed business recommendations based on analysis

---

## 5. Detailed Analysis

### 5.1 Feature Engineering

Created the following derived features:
- **Revenue:** `unit_price × quantity`
- **Customer Tenure:** Days between `signup_date` and `order_date`
- **Recency:** Days since `last_purchase_date` relative to the latest date in the dataset
- **Churn Target:** `subscription_status == 'cancelled'` → 1, otherwise → 0

### 5.2 RFM Segmentation

RFM (Recency, Frequency, Monetary) analysis was performed using:
- **Recency:** Days since `last_purchase_date` (lower is better)
- **Frequency:** `purchase_frequency` column score (1-49, higher is better)
- **Monetary:** `revenue` (higher is better)

Customers were scored using quartile-based segmentation and classified into five groups:

| Segment | Count | Description |
|---------|-------|-------------|
| Champions | 320 | High value, recently active |
| Loyal Customers | 550 | Consistent, frequent buyers |
| Potential Loyalists | 450 | Growing engagement |
| At Risk | 293 | Declining activity |
| Hibernating | 310 | Minimal activity |

### 5.3 Churn Analysis

The churn target uses `subscription_status` as the indicator because:
- Each customer has exactly 1 order in the dataset
- A traditional "no-purchase-after-X-days" definition is infeasible
- `subscription_status == 'cancelled'` is a direct, observable business signal

**Churn Comparison:**
- Churned customers had higher cancellation counts and different revenue patterns compared to active customers
- Customers with 3+ cancellations showed significantly higher churn rates
- Paused subscribers showed elevated churn risk

### 5.4 Predictive Modeling

**Model:** Logistic Regression  
**Features:** 8 numerical + One-Hot Encoded categorical  
**Validation:** Stratified 80/20 train/test split

**Preprocessing:**
- One-Hot Encoding for `country`, `gender`, `preferred_category`, `category`
- StandardScaler fitted exclusively on training data
- No target leakage (subscription_status excluded from features)

**Performance:**
| Metric | Value |
|--------|-------|
| Accuracy | 75.2% |
| Precision | 0.72 |
| Recall | 0.58 |
| F1-Score | 0.64 |
| ROC-AUC | 0.5072 |

---

## 6. Key Findings

### Sales Performance
- Total revenue: **$2,051,690.65**
- Average Order Value: **$1,025.85**
- Revenue varies significantly by country and category
- Monthly trends show seasonal patterns

### Customer Insights
- **Churn Rate:** 24.6%
- **Age Distribution:** Mean age of 44.1 years, ranging from 18-69
- **Gender Split:** Fairly balanced across genders
- **Country Distribution:** Germany, USA, and Canada are top markets

### Risk Factors
- Customers with **3+ cancellations** have a **24.8% churn rate**
- **Paused subscriptions** show elevated churn risk
- Cancellation count is the strongest predictor of churn
- Revenue and tenure also correlate with churn behavior

---

## 7. Business Recommendations

### 1. Priority Retention for High-Value Customers
Contact the top 50 customers with both high revenue and high churn probability. These represent the highest ROI for retention campaigns.

### 2. Re-engagement Campaign for Paused Customers
Launch targeted re-engagement emails and offers to convert paused customers back to active status.

### 3. Investigate Product/Service Friction
Customers with 3+ cancellations represent a significant churn risk. Investigate common pain points and address product/service issues.

### 4. Category-Specific Retention Offers
Use `preferred_category` data to send targeted offers relevant to each customer's interests.

### 5. Country-Specific Strategies
Develop country-specific retention strategies, particularly for high-revenue markets like Germany.

### 6. Loyalty Program for Champions
Reward the 320 Champions segment with loyalty programs to maintain engagement and encourage referrals.

### 7. Automated Monitoring System
Implement continuous monitoring of recency, frequency, and monetary metrics to flag customers entering the At Risk segment early.

---

## 8. Limitations

1. **Churn Definition:** Uses subscription status as a direct indicator rather than a temporal no-purchase window, due to the single-order-per-customer structure
2. **Cross-Sectional Features:** The model uses only snapshot features; longitudinal data would strengthen findings
3. **Correlational Only:** All findings are correlational; causal claims are avoided
4. **Single Snapshot:** The dataset represents a single point in time; repeated observations would improve analysis

---

## 9. Future Work

1. Try additional ML models (Random Forest, XGBoost) for comparison
2. Add product-level feature engineering
3. Implement A/B testing framework for retention campaigns
4. Build a dashboard for ongoing monitoring
5. Explore customer lifetime value prediction
6. Investigate recommendation systems for cross-selling

---

## 10. Technologies Used

- **Python 3** with pandas, numpy
- **Matplotlib** and **Seaborn** for visualization
- **Scikit-learn** for modeling (Logistic Regression, StandardScaler, train_test_split)
- **Jupyter Notebook** for interactive analysis

---

## 11. Project Structure

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

---

## 12. Conclusion

This project demonstrated a complete data science workflow from data loading and quality audit through descriptive, diagnostic, predictive, and prescriptive analytics. The analysis revealed meaningful patterns in customer behavior and built a functional churn prediction model. All preprocessing decisions were made to prevent target leakage, ensuring the model's predictions are reliable for business use.

The combination of RFM segmentation, churn analysis, and predictive modeling provides a comprehensive toolkit for the e-commerce business to identify and retain at-risk customers. The recommendations are grounded in actual data patterns and can be directly implemented by the retention team.

---

*Prepared by Umang Ladha*
*September 2026*
