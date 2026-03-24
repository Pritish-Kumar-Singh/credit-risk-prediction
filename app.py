import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -------------------------------
# LOAD MODELS
# -------------------------------
lgbm = joblib.load("models/lgbm.pkl")
rf = joblib.load("models/rf.pkl")
log_model = joblib.load("models/log.pkl")
scaler = joblib.load("models/scaler.pkl")
explainer = joblib.load("models/explainer.pkl")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

# -------------------------------
# TITLE
# -------------------------------
st.title("💳 AI Credit Default Prediction System")
st.markdown("### Predict loan default risk with AI + Explainability")

st.markdown("""
### 🧠 How this works
This system uses Machine Learning models (LightGBM, Random Forest, Logistic Regression)
to predict whether a customer is likely to default on credit payments.

It also provides explainability using SHAP to understand **why** the prediction was made.
""")

# -------------------------------
# INPUT SECTION
# -------------------------------
st.sidebar.header("📥 Customer Financial Profile")

st.sidebar.markdown("Fill in customer details to assess default risk")

credit_limit = st.sidebar.number_input(
    "💳 Credit Limit",
    help="Maximum credit amount assigned to the customer",
    value=200000
)

gender = st.sidebar.selectbox(
    "👤 Gender",
    options=["Male", "Female"]
)

education = st.sidebar.selectbox(
    "🎓 Education Level",
    options=[
        "Graduate School",
        "University",
        "High School",
        "Others"
    ],
    help="Customer’s highest education qualification"
)

marital = st.sidebar.selectbox(
    "💍 Marital Status",
    options=[
        "Married",
        "Single",
        "Others"
    ]
)

age = st.sidebar.number_input(
    "🎂 Age",
    help="Customer age in years",
    value=30
)

st.sidebar.markdown("### 💰 Financial Behavior")

avg_delay = st.sidebar.number_input(
    "⏳ Average Payment Delay",
    help="Average delay in months for past payments (0 = on time)",
    value=1.0
)

delay_count = st.sidebar.number_input(
    "⚠️ Number of Delayed Payments",
    help="Total number of times payments were delayed",
    value=2
)

total_bill = st.sidebar.number_input(
    "🧾 Total Bill Amount",
    help="Total outstanding bill amount",
    value=50000
)

total_payment = st.sidebar.number_input(
    "💵 Total Payment Made",
    help="Total amount paid by the customer",
    value=20000
)

# -------------------------------
# FEATURE ENGINEERING (IMPORTANT)
# -------------------------------
payment_ratio = total_payment / (total_bill + 1)
utilization_ratio = total_bill / (credit_limit + 1)
payment_consistency = total_payment / (credit_limit + 1)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.sidebar.button("🚀 Predict"):

    # ✅ ADD HERE (RIGHT AFTER BUTTON CLICK)

    gender = 1 if gender == "Male" else 2

    education_map = {
        "Graduate School": 1,
        "University": 2,
        "High School": 3,
        "Others": 4
    }
    education = education_map[education]

    marital_map = {
        "Married": 1,
        "Single": 2,
        "Others": 3
    }
    marital = marital_map[marital]

    input_data = pd.DataFrame([{
        'credit_limit': credit_limit,
        'gender': gender,
        'education_level': education,
        'marital_status': marital,
        'age': age,
        'avg_delay': avg_delay,
        'delay_count': delay_count,
        'total_bill': total_bill,
        'total_payment': total_payment,
        'payment_ratio': payment_ratio,
        'utilization_ratio': utilization_ratio,
        'payment_consistency': payment_consistency
    }])

    # -------------------------------
    # SCALE FOR LOGISTIC
    # -------------------------------
    input_scaled = scaler.transform(input_data)

    # -------------------------------
    # MODEL PREDICTIONS
    # -------------------------------
    lgb_prob = lgbm.predict_proba(input_data)[:,1][0]
    rf_prob = rf.predict_proba(input_data)[:,1][0]
    log_prob = log_model.predict_proba(input_scaled)[:,1][0]

    # -------------------------------
    # ENSEMBLE
    # -------------------------------
    final_prob = (0.7 * lgb_prob) + (0.2 * rf_prob) + (0.1 * log_prob)
    prediction = 1 if final_prob > 0.4 else 0

    # -------------------------------
    # OUTPUT SECTION
    # -------------------------------
    st.subheader("🔮 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Default Probability", f"{final_prob:.2f}")

    with col2:
        if prediction == 1:
            st.error("⚠️ High Risk of Default")
        else:
            st.success("✅ Low Risk Customer")

    # -------------------------------
    # SHAP EXPLANATION
    # -------------------------------
    st.subheader("🧠 Why this prediction? (Explainable AI)")

    shap_values = explainer(input_data)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # -------------------------------
    # FEATURE IMPORTANCE BAR
    # -------------------------------
    st.subheader("📊 Feature Impact")

    fig2, ax2 = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig2)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning, SHAP & Streamlit")