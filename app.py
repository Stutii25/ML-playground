# main_app.py

import streamlit as st
from supervised2 import supervised_main
from clustering2 import unsupervised_main

# Streamlit page settings
st.set_page_config(page_title="ML Visualizer", layout="wide", page_icon="📊")

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.selectbox("Choose Section:", [
    "🏠 Home", "🧠 Supervised Learning", "🔎 Unsupervised Learning"
])

# Home Page
if app_mode == "🏠 Home":
    st.markdown("""
        <h1 style='text-align:center; color:#2e86de;'>ML PlayGround: Visualise. Learn. Apply.</h1>
        <hr>
        <p style='text-align:center;'>Welcome to the ultimate machine learning visual tool combining both Supervised and Unsupervised learning!</p>
        <br>
        <h3>🧠 Supervised Learning Includes:</h3>
        <ul>
            <li>Logistic Regression</li>
            <li>Decision Tree</li>
            <li>Random Forest</li>
            <li>SVM</li>
            <li>Linear Regression</li>
        </ul>
        <h3>🔎 Unsupervised Learning Includes:</h3>
        <ul>
            <li>K-Means Clustering</li>
            <li>Hierarchical Clustering</li>
            <li>DBSCAN</li>
        </ul>
        <hr>
        <p style='text-align:center;'>📬 Created by <strong>Stuti Agrawal</strong> | 📧 stutiagrawal61@gmail.com</p>
    """, unsafe_allow_html=True)

# Supervised Page
elif app_mode == "🧠 Supervised Learning":
    supervised_main()

# Unsupervised Page
elif app_mode == "🔎 Unsupervised Learning":
    unsupervised_main()
