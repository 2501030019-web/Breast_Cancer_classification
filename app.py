import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Breast Cancer AI", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* REMOVE SIDEBAR */
section[data-testid="stSidebar"] {
    display: none;
}

/* FULL DARK MEDICAL GRADIENT BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #1a0d1f, #000000);
    background-attachment: fixed;
}

/* TITLE */
.title {
    font-size: 65px;
    font-weight: bold;
    text-align: center;
    color: #ff2e88;
    margin-top: 40px;
}

.subtitle {
    text-align: center;
    color: #dddddd;
    font-size: 22px;
    margin-bottom: 50px;
}

/* NAVIGATION BAR */
.navbar {
    display: flex;
    justify-content: center;
    gap: 50px;
    padding: 20px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    margin-top: 20px;
}

/* GLASS CARD */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,46,136,0.3);
    backdrop-filter: blur(15px);
    padding: 40px;
    border-radius: 25px;
    margin: 30px auto;
    width: 85%;
    transition: 0.4s;
}

.card:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 40px rgba(255,46,136,0.4);
}

/* STAT BOX */
.stat {
    background: rgba(255,46,136,0.1);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    font-size: 22px;
    color: white;
    border: 1px solid rgba(255,46,136,0.4);
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(45deg, #ff2e88, #ff6bb5);
    color: white;
    border-radius: 30px;
    padding: 10px 25px;
    border: none;
    font-size: 16px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.1);
}

.stApp {
    background-image: url("https://images.pexels.com/photos/7089401/pexels-photo-7089401.jpeg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.stApp::before {
    content:"";
    position:fixed;
    top:0; left:0;
    width:100%; height:100%;
    background: rgba(0,0,0,0.96);
    z-index:-1;
}
</style>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"
with col2:
    if st.button("📊 Dataset"):
        st.session_state.page = "Dataset"
with col3:
    if st.button("🔮 Prediction"):
        st.session_state.page = "Prediction"
with col4:
    if st.button("📈 Visualization"):
        st.session_state.page = "Visualization"

# ---------------- TITLE ----------------
st.markdown('<div class="title">🎗 Breast Cancer Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Early Tumor Classification System</div>', unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------- HOME ----------------
if st.session_state.page == "Home":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="stat">📊 30+ Features</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat">⚡ {round(accuracy*100,2)}% Accuracy</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat">🤖 Logistic Regression Model</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🩺 About This Project")
    st.write("""
    This AI system helps in early detection of breast cancer tumors 
    using machine learning classification techniques.
    
    ✔ Fast prediction  
    ✔ Accurate model  
    ✔ Medical focused UI  
    ✔ Real-time data processing  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- DATASET ----------------
elif st.session_state.page == "Dataset":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📊 Dataset Preview")
    st.dataframe(X.head())
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
elif st.session_state.page == "Prediction":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🔮 Upload CSV for Prediction")

    uploaded_file = st.file_uploader("Upload CSV file (30 features required)", type=["csv"])

    if uploaded_file:
        user_data = pd.read_csv(uploaded_file)
        prediction = model.predict(user_data)
        user_data["Prediction"] = prediction
        user_data["Prediction"] = user_data["Prediction"].map(
            {0: "Malignant", 1: "Benign"})

        st.success("Prediction Completed Successfully!")
        st.dataframe(user_data)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- VISUALIZATION ----------------
elif st.session_state.page == "Visualization":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📈 Feature Importance")

    importance = pd.Series(model.coef_[0], index=X.columns)
    importance.sort_values().plot(kind='barh', figsize=(8,10))
    st.pyplot(plt)

    st.markdown('</div>', unsafe_allow_html=True)
