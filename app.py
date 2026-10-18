
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="RetailIQ AI",
    page_icon="🛍️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

saved = load_model()
model = saved["model"]
features = saved["features"]

# STEP 31: Application Navigation

page = st.sidebar.radio(
    "Navigation",
    [
        "🤖 AI Prediction",
        "📊 Business Dashboard"
    ]
)
if page == "🤖 AI Prediction":
    st.title("🛍️ RetailIQ AI")
    st.subheader("Customer Purchase Behavior Prediction")

    st.write(
        "Predict the likelihood that a customer will not "
        "make another purchase within the next 90 days."
    )

    st.divider()

    st.header("Enter Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input(
            "Days Since Last Purchase",
            min_value=0,
            max_value=1000,
            value=30
        )

        frequency = st.number_input(
            "Total Previous Orders",
            min_value=1,
            value=5
        )

        monetary = st.number_input(
            "Total Customer Spending (£)",
            min_value=0.0,
            value=1500.0
        )

    with col2:
        total_items = st.number_input(
            "Total Items Purchased",
            min_value=0,
            value=100
        )

        tenure = st.number_input(
            "Customer History (Days)",
            min_value=0,
            max_value=2000,
            value=365
        )

    if st.button("🔮 Predict Customer Behavior"):

        input_data = pd.DataFrame([{
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "Total_Items": total_items,
            "Tenure_Days": tenure
        }])[features]

        probability = model.predict_proba(input_data)[0][1]

        st.divider()

        st.subheader("Prediction Result")

        st.metric(
            "Estimated 90-Day Non-Return Probability",
            f"{probability * 100:.1f}%"
        )

        if probability >= 0.7:
            st.error("High predicted non-return risk")

        elif probability >= 0.4:
            st.warning("Moderate predicted non-return risk")

        else:
            st.success("Low predicted non-return risk")

        st.progress(float(probability))

        st.caption(
            "This is a historical-data-based estimate, "
            "not a guarantee of future customer behavior. "
            "Risk bands are illustrative, not calibrated."
        )


# Business Dashboard
if page == "📊 Business Dashboard":
    st.title("📊 RetailIQ AI - Business Dashboard")

    st.write(
        "Explore retail sales, customer behavior, "
        "and machine learning insights."
    )

    st.success("Business Dashboard loaded successfully!")

    # Load deployment datasets
    @st.cache_data
    def load_business_summary():
        return pd.read_csv("deployment_data/business_summary.csv")

    @st.cache_data
    def load_customer_segments():
        return pd.read_csv("deployment_data/segment_counts.csv")

    business_summary = load_business_summary().iloc[0]

    # Business Performance Overview
    st.subheader("📈 Business Performance Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Revenue",
            f"£{business_summary['Total_Revenue']:,.2f}"
        )

    with col2:
        st.metric(
            "Total Transactions",
            f"{business_summary['Total_Transactions']:,.0f}"
        )

    with col3:
        st.metric(
            "Total Customers",
            f"{business_summary['Total_Customers']:,.0f}"
        )

    # Monthly Revenue Analysis
    st.divider()
    st.subheader("📈 Monthly Revenue Trends")

    monthly_revenue = pd.read_csv(
        "deployment_data/monthly_revenue.csv",
        parse_dates=["Month"]
    )

    st.line_chart(
        monthly_revenue.set_index("Month")["Revenue"]
    )

    # Country Revenue Analysis
    st.divider()
    st.subheader("🌍 Top 10 Countries by Revenue")

    country_revenue = (
        pd.read_csv("deployment_data/country_revenue.csv")
        .set_index("Country")["Revenue"]
    )

    st.bar_chart(
        country_revenue,
        horizontal=True,
        x_label="Country",
        y_label="Revenue (£)"
    )

    # Customer Segmentation Analysis
    st.divider()
    st.subheader("👥 Customer Segmentation")

    segment_counts_df = load_customer_segments()

    st.write("Customer purchasing behavior overview")

    st.metric(
        "Customers Analyzed",
        f"{segment_counts_df['Count'].sum():,}"
    )

    segment_counts = (
        segment_counts_df.set_index("Segment")["Count"]
    )

    st.bar_chart(segment_counts)

    # Machine Learning Performance Dashboard
    st.divider()
    st.subheader("🤖 Machine Learning Performance")

    st.write(
        "Historical holdout performance of the Random Forest "
        "model for predicting 90-day customer non-return."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "72.28%")
        st.metric("Precision", "69.51%")

    with col2:
        st.metric("Recall", "91.99%")
        st.metric("F1 Score", "79.18%")

    with col3:
        st.metric("ROC-AUC", "79.41%")

    st.info(
        "Metrics are from the September 2011 historical "
        "holdout evaluation, not an independent test of "
        "the final model trained on all snapshots."
    )

    # Machine Learning Model Comparison
    st.divider()
    st.subheader("📊 Machine Learning Model Comparison")

    comparison_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Logistic Regression": [
            71.92, 69.49, 90.92, 78.77, 78.75
        ],
        "Random Forest": [
            72.28, 69.51, 91.99, 79.18, 79.41
        ]
    })

    st.bar_chart(
        comparison_df,
        x="Metric",
        y=["Logistic Regression", "Random Forest"],
        stack=False,
        y_label="Performance (%)"
    )

    st.caption(
        "Comparison based on historical holdout evaluation results."
    )
