import streamlit as st
import pandas as pd
import pickle

# Load Model and Scaler
model = pickle.load(open("breast_cancer_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Tumor Prediction Page")

st.write("Upload a CSV file containing tumor diagnostic features")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data")
    st.write(data.head())

    # Required columns
    cols = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

    if st.button("Predict"):
        
        data[cols] = scaler.transform(data[cols])
        prediction = model.predict(data[cols])

        data["Prediction"] = prediction
        data["Prediction"] = data["Prediction"].map({0: "Benign", 1: "Malignant"})

        st.write("### Prediction Results")
        st.write(data)

        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "prediction_results.csv",
            "text/csv"
        )
