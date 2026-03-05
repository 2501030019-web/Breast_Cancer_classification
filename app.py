import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Breast Cancer AI", layout="wide")

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "users" not in st.session_state:
    st.session_state.users = {}
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1581093458791-9f3c3900df4b");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}
.stApp::before {
    content:"";
    position:fixed;
    top:0; left:0;
    width:100%; height:100%;
    background: rgba(0,0,0,0.92);
    z-index:-1;
}
.card {
    background:rgba(255,255,255,0.05);
    padding:30px;
    border-radius:20px;
    backdrop-filter: blur(15px);
    border:1px solid rgba(255,46,136,0.3);
    margin-bottom:30px;
}
.title {
    font-size:50px;
    color:#ff2e88;
}
.footer {
    margin-top:80px;
    padding:30px;
    background:rgba(0,0,0,0.8);
    text-align:center;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
left, right = st.columns([1,3])

# ---------------- LEFT SIDE LOGIN PANEL ----------------
with left:
    st.markdown("## 🎗 User Panel")

    if not st.session_state.logged_in:

        option = st.radio("Select", ["Login", "Sign Up"])

        if option == "Login":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if username in st.session_state.users and st.session_state.users[username] == password:
                    st.session_state.logged_in = True
                    st.success("Login Successful")
                    st.rerun()
                else:
                    st.error("Invalid Credentials")

        else:
            new_user = st.text_input("Create Username")
            new_pass = st.text_input("Create Password", type="password")

            if st.button("Create Account"):
                st.session_state.users[new_user] = new_pass
                st.success("Account Created! Please Login.")

    else:
        st.success("Logged In")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

# ---------------- RIGHT SIDE MAIN CONTENT ----------------
with right:

    if st.session_state.logged_in:

        # Navigation
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🏠 Home"):
                st.session_state.page = "Home"
        with col2:
            if st.button("📊 About Dataset"):
                st.session_state.page = "Dataset"
        with col3:
            if st.button("🔮 Prediction"):
                st.session_state.page = "Prediction"
        with col4:
            if st.button("📈 Visualization"):
                st.session_state.page = "Visualization"

        st.markdown("<div class='title'>Breast Cancer AI Detection</div>", unsafe_allow_html=True)
        st.markdown("---")

        # Load Model
        model = pickle.load(open("breast_cancer_model.pkl","rb"))
        scaler = pickle.load(open("scaler.pkl","rb"))

        # HOME
        if st.session_state.page == "Home":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.header("About Project")
            st.write("AI powered system for early breast cancer detection.")
            st.success(f"Model Accuracy: {round(accuracy*100,2)}%")
            st.markdown("</div>", unsafe_allow_html=True)

        # DATASET
        elif st.session_state.page == "Dataset":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.header("Dataset Preview")
            st.dataframe(X.head())
            st.markdown("</div>", unsafe_allow_html=True)

        # PREDICTION
        elif st.session_state.page == "Prediction":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.header("Upload CSV for Prediction")
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            cols = ['radius_mean', 'texture_mean', 'perimeter_mean',
               'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
               'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
               'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
               'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
               'fractal_dimension_se', 'radius_worst', 'texture_worst',
               'perimeter_worst', 'area_worst', 'smoothness_worst',
               'compactness_worst', 'concavity_worst', 'concave points_worst',
               'symmetry_worst', 'fractal_dimension_worst']

            if uploaded_file:
                user_data = pd.read_csv(uploaded_file)
                scaled_data = scaler.transform(user_data)
                prediction = model.predict(scaled_data)
                user_data["Prediction"] = prediction
                user_data["Prediction"] = user_data["Prediction"].map(
                    {0: "Malignant", 1: "Benign"})
                st.dataframe(user_data)

            st.markdown("</div>", unsafe_allow_html=True)

        # VISUALIZATION
        elif st.session_state.page == "Visualization":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.header("Feature Importance")
            importance = pd.Series(model.coef_[0], index=X.columns)
            importance.sort_values().plot(kind='barh', figsize=(8,8))
            st.pyplot(plt)
            st.markdown("</div>", unsafe_allow_html=True)

        # FOOTER
        st.markdown("""
        <div class="footer">
        <h3>📞 Contact Us</h3>
        <p>Email: support@breastcancerai.com</p>
        <p>Phone: +91 9876543210</p>
        <p>© 2026 Breast Cancer AI</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Please Login to Access the Dashboard")
