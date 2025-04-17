import streamlit as st
from auth import show_auth_page

# Page Configuration
st.set_page_config(
    page_title="Golf Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
        .big-font { font-size:32px !important; font-weight: bold; }
        .medium-font { font-size:20px !important; }
        .highlight-box { 
            background-color: #f0f2f6; 
            padding: 15px; 
            border-radius: 10px; 
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Check if user is logged in
if 'user_id' not in st.session_state or not st.session_state.user_id:
    show_auth_page()
    st.stop()

# Header and Introduction
st.header("🏌️‍♂️ Golf Analytics Dashboard")
st.markdown('<p class="big-font">Welcome to your personalized golf improvement hub! ⛳</p>', unsafe_allow_html=True)

st.markdown("""
This cutting-edge dashboard is designed to elevate your golf performance using advanced analytics, 
machine learning, and geospatial shot predictions. With insights powered by AI, you'll receive 
tailored coaching to refine your swing, optimize ball trajectory, and maximize accuracy. 🎯
""")

# Golf Analytics Features
st.markdown("""
### 🔍 What You Can Do Here:
- **🏌️ Virtual Golf Coach** – Get personalized swing analysis and actionable feedback using **NLP-powered AI**.
- **🌎 Geospatial Shot Mapping** – Visualize your shots on an interactive map and predict landing zones.
- **📊 AI-Powered Insights** – Leverage data-driven recommendations based on research from **Golf Galaxy's dataset**.
- **📈 Performance Trends** – Track your improvement over time with historical shot analytics.
""")

# Highlight Section
st.markdown("""
### 📢 How It Works:
1. **Log Your Shots** – Upload or enter your shot data.
2. **Receive Expert Feedback** – Our AI model evaluates your performance.
3. **Explore Geospatial Predictions** – See where your shots are likely to land.
4. **Improve with Data-Driven Insights** – Get recommendations based on professional research.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Call to Action
st.markdown("""
Ready to enhance your game? Start by logging your shots and asking the **Virtual Coach** a question! 🎯
""")

# Logout button
if st.button("Logout"):
    st.session_state.user_id = None
    st.rerun()
