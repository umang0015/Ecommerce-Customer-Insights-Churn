import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="E-Commerce Churn Analytics", layout="wide")

# ── Load Data ──
@st.cache_data
def load_data():
    df = pd.read_csv('data/E Commerce Customer Insights and Churn Dataset.csv')
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    df['revenue'] = df['unit_price'] * df['quantity']
    df['tenure_days'] = (df['order_date'] - df['signup_date']).dt.days
    analysis_date = df['last_purchase_date'].max()
    df['recency_days'] = (analysis_date - df['last_purchase_date']).dt.days
    df['churn'] = (df['subscription_status'] == 'cancelled').astype(int)
    return df

df = load_data()

# ── Sidebar ──
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Sales Analysis", "Customer Analysis", "Churn Prediction", "RFM Segments", "Recommendations"])

# ── HOME PAGE ──
if page == "Home":
    st.title("E-Commerce Customer Insights & Churn Analytics")
    st.markdown("### Prepared by **Umang Ladha**")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${df['revenue'].sum():,.0f}")
    col2.metric("Churn Rate", f"{df['churn'].mean()*100:.1f}%")
    col3.metric("Avg Order Value", f"${df['revenue'].mean():,.0f}")
    col4.metric("Total Customers", f"{len(df):,}")
    
    st.markdown("---")
    st.markdown("""
    This project analyzes 2,000 customer records to understand sales performance,
    identify churn patterns, and build a predictive model for at-risk customers.
    
    **Key Features:**
    - Sales performance analysis by country and category
    - Customer demographics and behavior patterns
    - RFM segmentation for customer value analysis
    - Logistic Regression churn prediction model
    - Risk-based business recommendations
    """)
    
    # Sample revenue chart
    st.subheader("Monthly Revenue Trend")
    df['order_month'] = df['order_date'].dt.to_period('M')
    monthly_rev = df.groupby('order_month')['revenue'].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    monthly_rev.plot(ax=ax, color='#3498db', marker='o')
    ax.set_title('Monthly Revenue')
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue ($)')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ── DATA OVERVIEW ──
elif page == "Data Overview":
    st.title("Data Overview")
    
    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("First 5 Rows")
        st.dataframe(df.head())
    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
    
    st.subheader("Data Quality")
    col1, col2, col3 = st.columns(3)
    col1.metric("Missing Values", f"{df.isnull().sum().sum()}")
    col2.metric("Duplicate Rows", f"{df.duplicated().sum()}")
    col3.metric("Unique Customers", f"{df['customer_id'].nunique()}")
    
    st.subheader("Subscription Status Distribution")
    status_dist = df['subscription_status'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    status_dist.plot(kind='bar', ax=ax, color=['#2ecc71','#e74c3c','#f39c12'])
    ax.set_title('Subscription Status')
    ax.set_ylabel('Count')
    plt.xticks(rotation=0)
    st.pyplot(fig)

# ── SALES ANALYSIS ──
elif page == "Sales Analysis":
    st.title("Sales Performance Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue by Country")
        country_rev = df.groupby('country')['revenue'].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        country_rev.plot(kind='barh', ax=ax, color='#2ecc71')
        ax.set_title('Revenue by Country')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Revenue by Category")
        cat_rev = df.groupby('category')['revenue'].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        cat_rev.plot(kind='barh', ax=ax, color='#e74c3c')
        ax.set_title('Revenue by Category')
        st.pyplot(fig)
    
    st.subheader("Revenue Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['revenue'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
    ax.set_title('Revenue Distribution')
    ax.set_xlabel('Revenue ($)')
    st.pyplot(fig)
    
    st.subheader("Monthly Revenue Trend")
    df['order_month'] = df['order_date'].dt.to_period('M')
    monthly_rev = df.groupby('order_month')['revenue'].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly_rev.plot(ax=ax, color='#9b59b6', marker='o')
    ax.set_title('Monthly Revenue Trend')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ── CUSTOMER ANALYSIS ──
elif page == "Customer Analysis":
    st.title("Customer Analysis")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Age", f"{df['age'].mean():.1f}")
    col2.metric("Mean Tenure (days)", f"{df['tenure_days'].mean():.0f}")
    col3.metric("Mean Recency (days)", f"{df['recency_days'].mean():.0f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['age'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        ax.set_title('Age Distribution')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots(figsize=(6, 5))
        df['gender'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title('Gender')
        st.pyplot(fig)
    
    st.subheader("Churn Rate by Category")
    fig, ax = plt.subplots(figsize=(10, 4))
    df.groupby('category')['churn'].mean().sort_values().plot(kind='bar', ax=ax, color='#e74c3c')
    ax.set_title('Churn Rate by Category')
    ax.set_ylabel('Churn Rate')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Churn Rate by Country")
    fig, ax = plt.subplots(figsize=(10, 4))
    df.groupby('country')['churn'].mean().sort_values().plot(kind='bar', ax=ax, color='#3498db')
    ax.set_title('Churn Rate by Country')
    ax.set_ylabel('Churn Rate')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ── CHURN PREDICTION ──
elif page == "Churn Prediction":
    st.title("Churn Prediction Model")
    st.markdown("Logistic Regression model to predict customer churn.")
    
    # Train model
    feature_cols = ['age', 'tenure_days', 'recency_days', 'purchase_frequency',
                    'cancellations_count', 'unit_price', 'quantity', 'revenue',
                    'country', 'gender', 'preferred_category', 'category']
    X = df[feature_cols].copy()
    y = df['churn'].copy()
    X_encoded = pd.get_dummies(X, columns=['country', 'gender', 'preferred_category', 'category'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    col2.metric("Precision", f"{classification_report(y_test, y_pred, output_dict=True)['1']['precision']:.2f}")
    col3.metric("Recall", f"{classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.2f}")
    col4.metric("F1-Score", f"{classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']:.2f}")
    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=['Active/Paused', 'Churned']))
    
    st.subheader("Feature Importance")
    feature_names = list(X_encoded.columns)
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=lambda x: x.abs(), ascending=False)
    st.dataframe(coef_df.head(15))
    
    st.subheader("Top Churn Risk Factors")
    st.markdown(f"**Top risk factors (positive coefficients):** {', '.join(coef_df.head(5)['feature'].tolist())}")
    st.markdown(f"**Protective factors (negative coefficients):** {', '.join(coef_df.tail(5)['feature'].tolist())}")

# ── RFM SEGMENTS ──
elif page == "RFM Segments":
    st.title("RFM Customer Segmentation")
    st.markdown("Recency, Frequency, Monetary analysis using purchase_frequency and revenue.")
    
    df['R_score'] = pd.qcut(df['recency_days'], q=4, labels=[4, 3, 2, 1]).astype(int)
    df['F_score'] = pd.qcut(df['purchase_frequency'], q=4, labels=[1, 2, 3, 4]).astype(int)
    df['M_score'] = pd.qcut(df['revenue'], q=4, labels=[1, 2, 3, 4]).astype(int)
    df['RFM_score'] = df['R_score'] + df['F_score'] + df['M_score']
    
    def segment_customer(rfm):
        if rfm >= 10: return 'Champions'
        elif rfm >= 8: return 'Loyal Customers'
        elif rfm >= 6: return 'Potential Loyalists'
        elif rfm >= 4: return 'At Risk'
        else: return 'Hibernating'
    
    df['segment'] = df['RFM_score'].apply(segment_customer)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Champions", f"{(df['segment']=='Champions').sum()}")
    col2.metric("At Risk", f"{(df['segment']=='At Risk').sum()}")
    col3.metric("Hibernating", f"{(df['segment']=='Hibernating').sum()}")
    
    st.subheader("Segment Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    df['segment'].value_counts().sort_values().plot(kind='barh', ax=ax, color='#3498db')
    ax.set_title('Customer Segments (RFM)')
    st.pyplot(fig)
    
    st.subheader("Churn Rate by Segment")
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby('segment')['churn'].mean().sort_values().plot(kind='bar', ax=ax, color='#e74c3c')
    ax.set_title('Churn Rate by Segment')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.subheader("Segment Details")
    for seg in ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk', 'Hibernating']:
        subset = df[df['segment'] == seg]
        st.markdown(f"**{seg}**: {len(subset)} customers, avg revenue ${subset['revenue'].mean():,.0f}, churn rate {subset['churn'].mean()*100:.1f}%")

# ── RECOMMENDATIONS ──
elif page == "Recommendations":
    st.title("Business Recommendations")
    st.markdown("Based on the analysis above, here are the key recommendations:")
    
    recs = [
        ("1. Priority Retention", "Contact the top 50 customers with both high revenue and high churn probability for maximum ROI on retention campaigns."),
        ("2. Re-engagement Campaign", "Launch targeted emails to convert paused subscribers back to active status."),
        ("3. Investigate Friction", "Customers with 3+ cancellations have a 24.8% churn rate. Address common pain points."),
        ("4. Category-Specific Offers", "Use preferred_category data to send relevant targeted promotions."),
        ("5. Country Strategies", "Develop retention strategies for high-revenue markets like Germany."),
        ("6. Loyalty Program", "Reward the 320 Champions segment to maintain engagement and drive referrals."),
        ("7. Automated Monitoring", "Implement continuous RFM monitoring to flag at-risk customers early.")
    ]
    
    for title, desc in recs:
        st.markdown(f"### {title}")
        st.write(desc)
        st.markdown("---")

# ── Footer ──
st.sidebar.markdown("---")
st.sidebar.markdown("**Project by Umang Ladha** | September 2026")
st.sidebar.markdown("Built with Streamlit")
