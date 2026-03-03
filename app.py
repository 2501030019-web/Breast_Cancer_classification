import streamlit as st

st.set_page_config(page_title="Breast Cancer App", layout="wide")

st.title("Breast Cancer Prediction System")

st.image("https://img.freepik.com/free-vector/breast-cancer-awareness-concept_23-2148574565.jpg", width=500)

st.markdown("""
## Welcome 👋

This application predicts whether a tumor is:

- ✅ Benign
- ❌ Malignant

Use the sidebar to navigate through different pages.
""")
