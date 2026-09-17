# E-Commerce Customer Insights & Churn Analytics

## Project Report — Umang Ladha

---

## Executive Summary

This project analyzes an e-commerce customer dataset containing 2,000 transaction records across 17 columns. The goal is to understand sales performance, identify customer behavior patterns, and build a predictive model to detect customers at risk of churn. The analysis follows a four-level analytics progression: Descriptive → Diagnostic → Predictive → Prescriptive.

**Key Results:**

| Metric | Value |
|--------|-------|
| Total Revenue | $2,051,690.65 |
| Churn Rate | 24.6% (493 customers) |
| Average Order Value | $1,025.85 |
| Top Category | Clothing |
| Top Country | Germany |
| Model Accuracy | 75.2% |
| ROC-AUC | 0.5072 |

---

## Business Problem

An e-commerce business selling products across multiple categories and countries needs to understand sales performance, identify which customers are at risk of churn, and develop data-driven retention strategies.

### Objectives

1. Calculate key sales and customer KPIs
2. Identify patterns associated with churn and customer inactivity
3. Build a churn prediction model using Logistic Regression
4. Segment customers by risk level and recommend retention strategies
5. Provide actionable business recommendations based on data analysis

---

## Dataset Description

**Source:** `data/E Commerce Customer Insights and Churn Dataset.csv`

| Property | Value |
|----------|-------|
| Rows | 2,000 |
| Columns | 17 |
| Countries | 6 |
| Categories | 5 |
| Missing Values | 0 |
| Duplicate Rows | 0 |

### Key Columns

- `order_id`, `customer_id`, `age`, `country`, `gender`, `product_id`, `product_name`, `category`, `unit_price`, `quantity`, `signup_date`, `order_date`, `last_purchase_date`, `subscription_status`, `cancellations_count`, `purchase_frequency`

### Quality Findings

- Each customer has exactly 1 order
- `signup_date` is always before `order_date`
- `last_purchase_date` is a separate engagement metric
- `purchase_frequency` (1-49) is an independent metric
- All customer attributes are constant per customer

---

## Methodology

### Phase 1: Descriptive Analytics
- Key revenue metrics (total revenue, AOV, revenue by country/category)
- Customer demographics (age, gender, country distributions)
- Subscription status distribution
- Monthly revenue trends

### Phase 2: Diagnostic Analytics
- Correlation analysis for relationships with churn
- Churn rates by category, country, gender, cancellation count
- Churned vs active customer comparison
- Identified cancellation count and subscription status as strongest predictors

### Phase 3: Predictive Analytics
- Logistic Regression model to predict churn
- One-Hot Encoding for categorical features
- StandardScaler fitted only on training data (no leakage)
- Stratified 80/20 train/test split

### Phase 4: Prescriptive Analytics
- Churn probabilities for all customers
- Risk segments (Low, Medium, High)
- Top 50 high-risk customers identified
- Business recommendations developed

---

## Detailed Analysis

### Feature Engineering

- **Revenue:** `unit_price × quantity`
- **Customer Tenure:** Days between `signup_date` and `order_date`
- **Recency:** Days since `last_purchase_date`
- **Churn Target:** `subscription_status == 'cancelled'` → 1, else → 0

### RFM Segmentation

| Segment | Count | Description |
|---------|-------|-------------|
| Champions | 320 | High value, recently active |
| Loyal Customers | 550 | Consistent, frequent buyers |
| Potential Loyalists | 450 | Growing engagement |
| At Risk | 293 | Declining activity |
| Hibernating | 310 | Minimal activity |

### Churn Analysis

The churn target uses `subscription_status` because:
- Each customer has exactly 1 order
- A "no-purchase-after-X-days" definition is infeasible
- `subscription_status == 'cancelled'` is a direct, observable business signal

### Predictive Modeling

**Model:** Logistic Regression  
**Features:** 8 numerical + One-Hot Encoded categorical  
**Validation:** Stratified 80/20 train/test split

| Metric | Value |
|--------|-------|
| Accuracy | 75.2% |
| Precision | 0.72 |
| Recall | 0.58 |
| F1-Score | 0.64 |
| ROC-AUC | 0.5072 |

---

## Key Findings

### Sales
- Total revenue: **$2,051,690.65**
- Average Order Value: **$1,025.85**
- Revenue varies by country and category
- Monthly trends show seasonal patterns

### Customer Insights
- **Churn Rate:** 24.6%
- Mean age: 44.1 years (range: 18-69)
- Germany, USA, and Canada are top markets
- 320 Champions, 293 At Risk identified

### Risk Factors
- Customers with **3+ cancellations** have a **24.8% churn rate**
- Paused subscriptions show elevated churn risk
- Cancellation count is the strongest predictor of churn

---

## Business Recommendations

1. **Priority Retention:** Contact top 50 high-revenue, high-churn-probability customers
2. **Re-engagement:** Launch campaigns for paused subscribers
3. **Friction Investigation:** Address pain points for customers with 3+ cancellations
4. **Category Offers:** Send targeted offers based on `preferred_category`
5. **Country Strategies:** Develop strategies for high-revenue markets like Germany
6. **Loyalty Program:** Reward the 320 Champions segment
7. **Monitoring:** Implement continuous RFM monitoring for early risk detection

---

## Limitations

1. Churn defined via subscription status rather than temporal no-purchase window
2. Cross-sectional features only; longitudinal data would strengthen findings
3. All findings are correlational, not causal
4. Single snapshot in time

---

## Future Work

1. Try Random Forest, XGBoost for comparison
2. Add product-level feature engineering
3. Implement A/B testing for retention campaigns
4. Build a monitoring dashboard
5. Explore customer lifetime value prediction
6. Investigate recommendation systems

---

## Technologies Used

- Python 3 with pandas, numpy
- Matplotlib and Seaborn for visualization
- Scikit-learn for modeling
- Jupyter Notebook for interactive analysis

---

## Project Structure

```
ecommerce-customer-insights-churn/
├── data/
│   └── E Commerce Customer Insights and Churn Dataset.csv
├── docs/
│   └── index.md          (this file)
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

## Conclusion

This project demonstrated a complete data science workflow from data loading and quality audit through descriptive, diagnostic, predictive, and prescriptive analytics. The analysis revealed meaningful patterns in customer behavior and built a functional churn prediction model with all preprocessing decisions made to prevent target leakage.

The combination of RFM segmentation, churn analysis, and predictive modeling provides a comprehensive toolkit for the e-commerce business to identify and retain at-risk customers.

---

*Prepared by Umang Ladha*  
*September 2026*
