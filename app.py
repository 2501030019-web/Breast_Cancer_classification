import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* REMOVE SIDEBAR */
section[data-testid="stSidebar"] {
    display: none;
}

/* DARK BACKGROUND IMAGE */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1588776814546-1ffcf47267a5");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0,0,0,0.85);
    z-index: -1;
}

/* TOP NAVIGATION */
.navbar {
    display: flex;
    justify-content: center;
    gap: 40px;
    padding: 15px;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(10px);
    border-radius: 15px;
}

.navbar button {
    background: none;
    border: none;
    color: white;
    font-size: 18px;
    cursor: pointer;
    transition: 0.3s;
}

.navbar button:hover {
    color: #ff4b7d;
    transform: scale(1.1);
}

/* TITLE */
.title {
    font-size: 55px;
    font-weight: bold;
    text-align: center;
    color: #ff4b7d;
    margin-top: 30px;
}

/* GLASS CARD */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    padding: 40px;
    border-radius: 20px;
    margin: 30px 0;
    transition: 0.4s;
}

.card:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 40px rgba(255,0,90,0.4);
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* FULL DARK MEDICAL BACKGROUND */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1530026186672-2cd00ffc50fe");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* EXTRA DARK OVERLAY */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.92);  /* More Dark */
    z-index: -1;
}

</style>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- TOP NAVIGATION ----------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"

with col2:
    if st.button("📊 About Dataset"):
        st.session_state.page = "About"

with col3:
    if st.button("🔮 Prediction"):
        st.session_state.page = "Prediction"

with col4:
    if st.button("📈 Visualization"):
        st.session_state.page = "Visualization"

# ---------------- TITLE ----------------
st.markdown('<div class="title">Breast Cancer Prediction System</div>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:white;'>AI Powered Tumor Classification (Benign vs Malignant)</h3>", unsafe_allow_html=True)

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

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📊 Dataset")
    st.write("Breast Cancer Wisconsin Dataset with 30 diagnostic features.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🤖 Model")
    st.write("Logistic Regression ML Model trained with high accuracy.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("⚡ Accuracy")
    st.success(f"Model Accuracy: {round(accuracy*100,2)}%")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ABOUT ----------------
elif st.session_state.page == "About":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Dataset Preview")
    st.dataframe(X.head())
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
elif st.session_state.page == "Prediction":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Upload CSV for Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        user_data = pd.read_csv(uploaded_file)
        prediction = model.predict(user_data)
        user_data["Prediction"] = prediction
        user_data["Prediction"] = user_data["Prediction"].map(
            {0: "Malignant", 1: "Benign"})

        st.success("Prediction Completed!")
        st.dataframe(user_data)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- VISUALIZATION ----------------
elif st.session_state.page == "Visualization":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Feature Importance")

    importance = pd.Series(model.coef_[0], index=X.columns)
    importance.sort_values().plot(kind='barh', figsize=(8,10))
    st.pyplot(plt)

    st.markdown('</div>', unsafe_allow_html=True)
