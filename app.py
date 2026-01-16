
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # New library for interactive charts
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Telco Analytics", page_icon="📊", layout="wide")

# --- 1. TRAIN MODEL & GENERATE DATA ---
@st.cache_resource
def load_data_and_model():
    np.random.seed(42)
    n_rows = 500  # Increased data for better charts
    data = {
        'Tenure': np.random.randint(1, 73, n_rows),
        'MonthlyCharges': np.random.uniform(30, 120, n_rows),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_rows),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_rows),
        'Churn': np.random.choice(['Yes', 'No'], n_rows, p=[0.3, 0.7])
    }
    df = pd.DataFrame(data)

    # Preprocessing for Model
    df_model = df.copy()
    mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2, 
               'DSL': 0, 'Fiber optic': 1, 'No': 2,
               'Yes': 1, 'No': 0}
    df_model.replace(mapping, inplace=True)

    X = df_model[['Tenure', 'MonthlyCharges', 'Contract', 'InternetService']]
    y = df_model['Churn']
    model = RandomForestClassifier()
    model.fit(X, y)

    return df, model

df_analytics, model = load_data_and_model()

# --- 2. AUTHENTICATION ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown(
        """<style>.login-box {padding: 2rem; border-radius: 10px; border: 1px solid #e0e0e0; background-color: #f9f9f9;}</style>""", 
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🔐 Analytics Portal")
        st.info("System Restricted to Authorized Personnel")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", type="primary"):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

# --- 3. MAIN DASHBOARD ---
def main_app():
    # Sidebar
    with st.sidebar:
        st.write(f"👤 **Admin User**")
        st.divider()
        menu = st.radio("Navigation", ["🔍 Prediction Engine", "📊 Data Insights"])
        st.divider()
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # TAB 1: PREDICTION
    if menu == "🔍 Prediction Engine":
        st.title("🔮 Churn Prediction Engine")
        st.write("Enter customer details to predict risk.")

        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Tenure (Months)", 0, 72, 24)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
        with col2:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        if st.button("Analyze Risk", type="primary"):
            # Map inputs to numbers
            c_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            i_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

            row = pd.DataFrame([[tenure, monthly_charges, c_map[contract], i_map[internet]]],
                               columns=['Tenure', 'MonthlyCharges', 'Contract', 'InternetService'])

            pred = model.predict(row)[0]
            prob = model.predict_proba(row)[0][1] * 100

            st.divider()
            if pred == 1:
                st.error(f"⚠️ HIGH RISK ({prob:.1f}%)")
                st.progress(int(prob))
                st.caption("Strategy: Offer 15% Discount immediately.")
            else:
                st.success(f"✅ SAFE CUSTOMER ({prob:.1f}%)")
                st.progress(int(prob))

    # TAB 2: INSIGHTS (CHARTS)
    elif menu == "📊 Data Insights":
        st.title("📈 Business Intelligence Dashboard")

        # Chart 1: Churn by Contract
        st.subheader("1. Which Contract Type Churns the Most?")
        churn_counts = df_analytics.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig1 = px.bar(churn_counts, x='Contract', y='Count', color='Churn', barmode='group',
                      color_discrete_map={'Yes': 'red', 'No': 'green'})
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Monthly Charges Distribution
        st.subheader("2. Does Price Affect Churn?")
        fig2 = px.histogram(df_analytics, x='MonthlyCharges', color='Churn', nbins=20,
                            color_discrete_map={'Yes': 'red', 'No': 'green'})
        st.plotly_chart(fig2, use_container_width=True)

# --- CONTROL FLOW ---
if st.session_state.logged_in:
    main_app()
else:
    login()
