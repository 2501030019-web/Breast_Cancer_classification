import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Breast Cancer AI", layout="wide")

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "users" not in st.session_state:
    st.session_state.users = {}

# ---------------- CSS ----------------
st.markdown("""
<style>

section[data-testid="stSidebar"] {display:none;}

.stApp {
    background-image: url("https://images.unsplash.com/photo-1581093458791-9f3c3900df4b");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}

.stApp::before {
    content:"";
    position:fixed;
    top:0; left:0;
    width:100%; height:100%;
    background: rgba(0,0,0,0.90);
    z-index:-1;
}

.title {
    text-align:center;
    font-size:60px;
    color:#ff2e88;
    margin-top:40px;
}

.card {
    background:rgba(255,255,255,0.05);
    padding:40px;
    border-radius:25px;
    backdrop-filter: blur(15px);
    width:70%;
    margin:40px auto;
    border:1px solid rgba(255,46,136,0.3);
}

.footer {
    margin-top:100px;
    padding:40px;
    background:rgba(0,0,0,0.8);
    text-align:center;
    color:white;
}

.stButton>button {
    background: linear-gradient(45deg,#ff2e88,#ff6bb5);
    color:white;
    border-radius:30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state.logged_in:

    st.markdown('<div class="title">🎗 Breast Cancer AI Portal</div>', unsafe_allow_html=True)

    option = st.radio("Select Option", ["Login", "Create Profile"], horizontal=True)

    if option == "Login":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid Credentials")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Create Profile"):
            st.session_state.users[new_user] = new_pass
            st.success("Profile Created! Please Login.")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN APP AFTER LOGIN ----------------
else:

    # NAVIGATION
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🏠 Home"):
            st.session_state.page = "Home"
    with col2:
        if st.button("🔮 Prediction"):
            st.session_state.page = "Prediction"
    with col3:
        if st.button("📈 Visualization"):
            st.session_state.page = "Visualization"
    with col4:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown('<div class="title">Breast Cancer Detection System</div>', unsafe_allow_html=True)

    # LOAD DATA
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    # HOME
    if st.session_state.page == "Home":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.header("About Project")
        st.write("AI Powered Early Tumor Detection System.")
        st.success(f"Model Accuracy: {round(accuracy*100,2)}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # PREDICTION
    elif st.session_state.page == "Prediction":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file:
            user_data = pd.read_csv(uploaded_file)
            prediction = model.predict(user_data)
            user_data["Prediction"] = prediction
            user_data["Prediction"] = user_data["Prediction"].map(
                {0: "Malignant", 1: "Benign"})
            st.dataframe(user_data)
        st.markdown('</div>', unsafe_allow_html=True)

    # VISUALIZATION
    elif st.session_state.page == "Visualization":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        importance = pd.Series(model.coef_[0], index=X.columns)
        importance.sort_values().plot(kind='barh', figsize=(8,10))
        st.pyplot(plt)
        st.markdown('</div>', unsafe_allow_html=True)

    # CONTACT US FOOTER
    st.markdown("""
    <div class="footer">
        <h3>📞 Contact Us</h3>
        <p>Email: support@breastcancerai.com</p>
        <p>Phone: +91 9876543210</p>
        <p>Location: India</p>
        <p>© 2026 Breast Cancer AI | All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)
