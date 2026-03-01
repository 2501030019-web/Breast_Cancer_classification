import streamlit as st
import numpy as np
import pickle

# ---------------- Page Config ---------------- #

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🧬",
    layout="wide"
)

# ---------------- Load Model ---------------- #

model = pickle.load(open("breast_cancer_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# ---------------- Sidebar ---------------- #

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to",["Home","Prediction"])

# ---------------- Home Page ---------------- #

if page == "Home":

    st.title("🧬 Breast Cancer Prediction System")

    st.write("""
    Welcome to the **Breast Cancer Prediction Web App**.

    This machine learning application predicts whether a tumor is  
    **Benign (Non-Cancerous)** or **Malignant (Cancerous)** using
    tumor diagnostic features.
    """)

    st.subheader("📌 Project Overview")

    st.write("""
    This project uses **Machine Learning classification** to analyze tumor features.

    **Technologies Used**
    - Python
    - Machine Learning
    - Scikit-learn
    - Streamlit
    """)

    st.subheader("⚙️ How It Works")

    st.write("""
    1. Enter tumor feature values
    2. Click **Predict**
    3. The model classifies the tumor
    """)

    st.success("Use the sidebar to go to the Prediction page")

# ---------------- Prediction Page ---------------- #

elif page == "Prediction":

    st.title("🔬 Tumor Prediction")

    st.write("Enter tumor feature values below")

    feature_names = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
    'compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
    'compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst'
    ]

    inputs = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):

        if i % 2 == 0:
            value = col1.number_input(feature)
        else:
            value = col2.number_input(feature)

        inputs.append(value)

    if st.button("Predict Tumor Type"):

        features = np.array([inputs])

        features = scaler.transform(features)

        prediction = model.predict(features)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("⚠ Malignant Tumor Detected")
        else:
            st.success("✅ Benign Tumor Detected")
