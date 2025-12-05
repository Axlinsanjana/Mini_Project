import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Fraud Detection", layout="wide")

# -----------------------------------------------------------
# CSS STYLING + BOOTSTRAP ICONS
# -----------------------------------------------------------
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">

<style>

/* ------------------------------
   LOGO
-------------------------------*/
.nav-logo {
    color: white !important;
    font-size: 28px;
    font-weight: 800;       
    text-align: center;
    background-color: #1e3a8a;
    padding: 16px;
    border-radius: 10px;
    width: 100%;
    display: block;
    margin: 0 auto 20px auto;
}

/* ------------------------------
   BLUE THEME TEXT
-------------------------------*/
html, body, [class*="css"], p, span, div, label, h1, h2, h3 {
    color: #1e3a8a !important;
}

/* ------------------------------
   NAV BUTTONS WITH ICONS
-------------------------------*/
.nav-buttons button {
    background-color: #1e3a8a !important;
    color: white !important;  
    font-weight: 800 !important;  
    padding: 12px 28px !important;
    border-radius: 10px !important;
    border: none !important;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
}

.nav-buttons button:hover {
    background-color: #334fb3 !important;
    color: white !important;
}

/* ------------------------------
   ALL BUTTONS (GENERAL)
-------------------------------*/
div.stButton > button,
div.stButton > button > div,
div.stButton > button span,
button[kind="primary"],
button[kind="primary"] > div,
button[kind="primary"] span {
    color: white !important;       
    font-weight: 800 !important;   
}

/* ------------------------------
   PREDICT BUTTONS ONLY (WHITE BG + BLUE TEXT)
-------------------------------*/
div.stButton > button[data-testid="predict_btn"],
div.stButton > button[data-testid="predict_another_btn"] {
    background-color: white !important;
    color: #1e3a8a !important;
    border: 2px solid #1e3a8a !important;
    border-radius: 10px !important;
}

div.stButton > button[data-testid="predict_btn"]:hover,
div.stButton > button[data-testid="predict_another_btn"]:hover {
    background-color: #f0f0f0 !important;
    color: #1e3a8a !important;
}

/* ------------------------------
   RESULT TEXT WHITE
-------------------------------*/
.result-box {
    color: white !important;
    font-weight: bold;
    font-size: 20px;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
}

/* ------------------------------
   FULL APP BACKGROUND
-------------------------------*/
.stApp {
    background-color: #e6f2ff;
}

/* ------------------------------
   DARK TABLE TEXT
-------------------------------*/
.dark-table table {
    color: #1e1e1e !important;
    border-collapse: collapse;
    width: 100%;
}
.dark-table th, .dark-table td {
    border: 1px solid #999;
    padding: 8px;
    text-align: center;
}
.dark-table th {
    background-color: #d9d9d9;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "input_data" not in st.session_state:
    st.session_state.input_data = None

# -----------------------------------------------------------
# NAVBAR COMPONENT WITH ICONS
# -----------------------------------------------------------
def navbar():
    st.markdown('<div class="navbar nav-buttons" style="margin-bottom:20px;">', unsafe_allow_html=True)

    # Logo
    st.markdown('<div class="nav-logo">E-Commerce Fraud Detection</div>', unsafe_allow_html=True)

    # Buttons with Bootstrap icons inside text
    col1, col2, col3, col4 = st.columns([1,1,1,1])

    with col1:
        if st.button("🏠 Home", key="home_btn", help="Go to Home", use_container_width=True):
            st.session_state.page = "Home"

    with col2:
        if st.button("📊 Prediction", key="pred_btn", help="Make a Prediction", use_container_width=True):
            st.session_state.page = "Prediction"

    with col3:
        if st.button("📋 Result", key="res_btn", help="View Result", use_container_width=True):
            st.session_state.page = "Result"

    with col4:
        if st.button("ℹ️ About", key="about_btn", help="About This App", use_container_width=True):
            st.session_state.page = "About"

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# LOAD MODEL + SCALER
# -----------------------------------------------------------
model = joblib.load("trained_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------------------------------------
# FEATURE / ENCODING MAPPINGS
# -----------------------------------------------------------
feature_names = [
    'Transaction Amount','Product Category','Quantity','Customer Age',
    'Customer Location','Device Used','Transaction Hour','AgeGroup',
    'Transaction_Year','Transaction_Month','Transaction_Day','Transaction_DOW',
    'Payment Method_bank transfer','Payment Method_credit card','Payment Method_debit card'
]

numeric_cols = [
    'Transaction Amount','Product Category','Quantity','Customer Age',
    'Customer Location','Device Used','Transaction Hour'
]

product_category_map = {"Electronics":0,"Clothing":1,"Books":2,"Home & Kitchen":3,"Other":4}
age_group_map = {"18-24":0,"25-34":1,"35-44":2,"45-54":3,"55-64":4,"65+":5}
customer_location_map = {"North":0,"South":1,"East":2,"West":3}
device_map = {"Mobile":0,"Desktop":1,"Tablet":2}

# -----------------------------------------------------------
# DISPLAY NAVBAR
# -----------------------------------------------------------
navbar()

# -----------------------------------------------------------
# PAGE LOGIC
# -----------------------------------------------------------

# HOME PAGE
if st.session_state.page == "Home":
    st.title("Welcome to the E-Commerce Fraud Detection System")
    st.write("This system predicts fraudulent transactions using Machine Learning.")
    st.image("https://res.cloudinary.com/dthpnue1d/image/upload/v1750151397/Top_5_Benefits_of_Implementing_a_Real_Time_Fraud_Detection_Agent_in_Your_Banking_App_Banner_5359dbf51d.webp")

# PREDICTION PAGE
elif st.session_state.page == "Prediction":
    st.title("Predict Fraudulent Transaction")

    amount = st.number_input("Transaction Amount", 0.0)
    qty = st.number_input("Quantity", 1)
    txn_hour = st.number_input("Transaction Hour", 0)

    product_category = st.selectbox("Product Category", list(product_category_map.keys()))
    age_group = st.selectbox("Age Group", list(age_group_map.keys()))
    cust_loc = st.selectbox("Customer Location", list(customer_location_map.keys()))
    device = st.selectbox("Device Used", list(device_map.keys()))
    txn_date = st.date_input("Transaction Date", datetime.today())
    year, month, day, dow = txn_date.year, txn_date.month, txn_date.day, txn_date.weekday()
    payment = st.selectbox("Payment Method", ["bank transfer","credit card","debit card"])

    input_data = pd.DataFrame([[0]*len(feature_names)], columns=feature_names)
    input_data.loc[0,'Transaction Amount'] = amount
    input_data.loc[0,'Product Category'] = product_category_map[product_category]
    input_data.loc[0,'Quantity'] = qty
    input_data.loc[0,'Customer Age'] = age_group_map[age_group]
    input_data.loc[0,'Customer Location'] = customer_location_map[cust_loc]
    input_data.loc[0,'Device Used'] = device_map[device]
    input_data.loc[0,'Transaction Hour'] = txn_hour
    input_data.loc[0,'AgeGroup'] = age_group_map[age_group]
    input_data.loc[0,'Transaction_Year'] = year
    input_data.loc[0,'Transaction_Month'] = month
    input_data.loc[0,'Transaction_Day'] = day
    input_data.loc[0,'Transaction_DOW'] = dow

    # Payment method OHE
    input_data['Payment Method_bank transfer'] = 1 if payment=="bank transfer" else 0
    input_data['Payment Method_credit card'] = 1 if payment=="credit card" else 0
    input_data['Payment Method_debit card'] = 1 if payment=="debit card" else 0

    # Scale numeric columns
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    st.session_state.input_data = input_data

    if st.button("Predict", key="predict_btn"):
        prediction = model.predict(input_data)[0]
        st.session_state.prediction_result = (
            "🚨 Fraudulent Transaction Detected!" if prediction==1 else "✔ Legitimate Transaction"
        )
        st.session_state.page = "Result"

# RESULT PAGE
elif st.session_state.page == "Result":
    st.title("Prediction Result")

    if st.session_state.prediction_result:
        if "Fraudulent" in st.session_state.prediction_result:
            st.markdown(f"""
                <div class="result-box" style="background-color:#b91c1c;">
                    {st.session_state.prediction_result}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box" style="background-color:#10b981;">
                    {st.session_state.prediction_result}
                </div>
            """, unsafe_allow_html=True)

    if st.session_state.input_data is not None:
        st.subheader("Transaction Details")
        st.markdown(
            f"<div class='dark-table'>{st.session_state.input_data.to_html(index=False)}</div>",
            unsafe_allow_html=True
        )

    if st.button("Predict Another Transaction", key="predict_another_btn"):
        st.session_state.page = "Prediction"
        st.session_state.prediction_result = None
        st.session_state.input_data = None

# ABOUT PAGE
# ABOUT PAGE
elif st.session_state.page == "About":
    st.title("About This App")
    st.markdown("""
    **E-Commerce Fraud Detection System** is a web-based Machine Learning application that predicts fraudulent transactions in real-time, helping businesses reduce financial losses and protect customers.

    **Key Features:**
    - **User-Friendly Interface:** Easy navigation with Home, Prediction, Result, and About pages.
    - **Machine Learning Prediction:** Detects fraudulent or legitimate transactions using pre-trained models like Logistic Regression or XGBoost.
    - **Feature Engineering:** Handles numeric features (amount, quantity, hour), categorical features (product category, payment method, device, customer location), and date-based features (year, month, day, weekday).
    - **Data Preprocessing:** Scales numeric data and one-hot encodes categorical features for accurate predictions.
    - **Results Visualization:** Color-coded prediction results (red = fraud, green = legitimate) and detailed transaction tables.
    - **Custom Styling & Icons:** Blue-themed UI with bold headings, buttons, and Bootstrap icons.
    - **Session State Management:** Keeps data persistent across pages for smooth navigation.

    **Use Case:** Ideal for e-commerce platforms, payment processors, or online marketplaces aiming for fast, reliable fraud detection.
    """)
