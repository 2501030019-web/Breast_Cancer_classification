import streamlit as st

st.title("Model Information")

st.markdown("""
## Machine Learning Model Used

We used:

- Logistic Regression
- Scikit-Learn

### Why Logistic Regression?

- Works well for binary classification
- Fast and efficient
- Gives high accuracy (~95%+)

### Model Performance

- Accuracy: 95–98%
- Evaluated using:
    - Confusion Matrix
    - Precision
    - Recall
    - F1 Score
""")