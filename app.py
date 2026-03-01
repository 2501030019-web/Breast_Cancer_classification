
import streamlit as st
import numpy as np
import pickle

# ---------------- Load Model ---------------- #

model = pickle.load(open("breast_cancer_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# ---------------- Sidebar ---------------- #

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home","Prediction"])

# ---------------- Home Page ---------------- #

if page == "Home":

    st.title("Breast Cancer Prediction System")

    st.write("""
    This Machine Learning web application predicts whether a tumor is **Benign** or **Malignant** 
    using medical tumor measurements.
    """)

    st.subheader("Project Information")

    st.write("""
    - Model Type: Machine Learning Classification
    - Algorithm: Logistic Regression / Random Forest
    - Dataset: Breast Cancer Dataset
    - Platform: Streamlit Cloud
    """)

    st.subheader("Features Used")

    st.write("""
    - Radius Mean  
    - Texture Mean  
    - Perimeter Mean  
    - Area Mean  
    """)

    st.success("Use the sidebar to go to Prediction page")


# ---------------- Prediction Page ---------------- #

if page == "Prediction":

    st.title("Tumor Prediction")

    st.write("Enter tumor measurements below")

    radius_mean = st.number_input("Radius Mean")
    texture_mean = st.number_input("Texture Mean")
    perimeter_mean = st.number_input("Perimeter Mean")
    area_mean = st.number_input("Area Mean")

    if st.button("Predict"):

        features = np.array([[radius_mean,texture_mean,perimeter_mean,area_mean]])

        features = scaler.transform(features)

        prediction = model.predict(features)

        if prediction[0] == 1:
            st.error("Malignant Tumor")
        else:
            st.success("Benign Tumor")
