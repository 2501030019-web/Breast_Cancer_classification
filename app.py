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
.stApp {
    background-image: url("https://images.unsplash.com/photo-1588776814546-1ffcf47267a5");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0,0,0,0.75);
    z-index: -1;
}
.title {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: #ff4b7d;
    animation: fadeIn 2s ease-in-out;
}
@keyframes fadeIn {
    0% {opacity:0; transform: translateY(-20px);}
    100% {opacity:1; transform: translateY(0);}
}
.card {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
    padding: 30px;
    border-radius: 20px;
    margin: 20px 0;
    transition: 0.4s;
    animation: slideUp 1.5s ease-in-out;
}
.card:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 30px rgba(255,0,90,0.5);
}
@keyframes slideUp {
    from {opacity:0; transform: translateY(50px);}
    to {opacity:1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🩺 Breast Cancer Prediction System</div>', unsafe_allow_html=True)
st.write("### AI Powered Tumor Classification (Benign vs Malignant)")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go To",
                        ["🏠 Home",
                         "📊 About Dataset",
                         "🔮 Prediction",
                         "📈 Visualization"])

# ---------------- LOAD DATA & TRAIN MODEL ----------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------- HOME PAGE ----------------
if page == "🏠 Home":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📊 Dataset")
    st.write("Breast Cancer Wisconsin Dataset with 30 diagnostic features.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🤖 Model")
    st.write("Logistic Regression Machine Learning model trained on real dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("⚡ Model Accuracy")
    st.success(f"Model Accuracy: {round(accuracy*100,2)}%")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("💡 Why This Project?")
    st.write("""
    • Early detection saves lives  
    • AI powered healthcare solution  
    • Real time prediction system  
    • User friendly web interface  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ABOUT DATASET ----------------
elif page == "📊 About Dataset":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Dataset Information")
    st.write(X.head())
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION PAGE ----------------
elif page == "🔮 Prediction":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Upload CSV for Prediction")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        user_data = pd.read_csv(uploaded_file)

        try:
            prediction = model.predict(user_data)
            user_data["Prediction"] = prediction
            user_data["Prediction"] = user_data["Prediction"].map(
                {0: "Malignant", 1: "Benign"})

            st.success("Prediction Completed!")
            st.dataframe(user_data)

        except:
            st.error("⚠ Please upload correct formatted dataset (30 features required).")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- VISUALIZATION PAGE ----------------
elif page == "📈 Visualization":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Feature Importance Visualization")

    importance = pd.Series(model.coef_[0], index=X.columns)
    importance.sort_values().plot(kind='barh', figsize=(8,10))
    st.pyplot(plt)

    st.markdown('</div>', unsafe_allow_html=True)
