import streamlit as st

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.title {
    font-size: 50px;
    font-weight: bold;
    color: #ff4b6e;
    text-align: center;
    animation: fadeIn 2s ease-in;
}

.subtitle {
    font-size: 22px;
    text-align: center;
    color: white;
}

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(255,75,110,0.3);
    margin-bottom: 20px;
}

@keyframes fadeIn {
    0% {opacity: 0;}
    100% {opacity: 1;}
}

</style>
""", unsafe_allow_html=True)

# ---------------- Hero Section ----------------

st.markdown('<div class="title">🩺 Breast Cancer Prediction System</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">AI Powered Tumor Classification (Benign vs Malignant)</div>', unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns(2)

with col1:
    st.image("/images/equipment.jpg", use_container_width=True)
with col2:
    st.image("/images/goggles.jpg", use_container_width=True)

st.write("")
st.write("")

# ---------------- Info Cards ----------------

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("""
    <div class="card">
    <h3 style='color:#ff4b6e;'>📊 Dataset</h3>
    <p>Breast Cancer Wisconsin Dataset with 30 diagnostic features.</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
    <h3 style='color:#ff4b6e;'>🤖 Model</h3>
    <p>Logistic Regression Machine Learning Model with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="card">
    <h3 style='color:#ff4b6e;'>⚡ Accuracy</h3>
    <p>Achieved ~95%+ prediction accuracy on test dataset.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ---------------- Footer ----------------

st.markdown("""
<hr>
<center style='color:gray'>
Developed by Ayush Krishna Sahoo | AI & ML Project
</center>
""", unsafe_allow_html=True)
