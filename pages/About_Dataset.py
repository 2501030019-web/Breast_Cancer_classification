import streamlit as st

st.title("About Breast Cancer Dataset")

st.markdown("""
## Dataset Information

The dataset used is:

Breast Cancer Wisconsin Dataset

It contains 30 numerical diagnostic features computed from digitized images of a breast mass.

### Target Variable:
- 0 → Benign
- 1 → Malignant

### Total Features:
30 Medical Features including:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
""")