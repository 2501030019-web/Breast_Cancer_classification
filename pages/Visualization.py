import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

st.title("Data Visualization")

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

st.write("### Target Distribution")

fig, ax = plt.subplots()
df["target"].value_counts().plot(kind="bar", ax=ax)
st.pyplot(fig)

st.write("0 = Benign, 1 = Malignant")
