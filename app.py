
import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("breast_cancer_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Breast Cancer Tumor Prediction")

st.write("Enter the tumor features below")

radius_mean = st.number_input("Radius Mean")
texture_mean = st.number_input("Texture Mean")
perimeter_mean = st.number_input("Perimeter Mean")
area_mean = st.number_input("Area Mean")

if st.button("Predict"):

    features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Malignant Tumor")
    else:
        st.success("Benign Tumor")
