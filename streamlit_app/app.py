import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Health Insurance Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🏥 Health Insurance Charges Prediction")
st.markdown("""
This application predicts US health insurance Charges (annual) based on various factors such as age, BMI, 
smoking status, and other relevant features using machine learning.
""")

# Main content placeholder
st.header("Welcome to the Health Insurance Charges Prediction App")
st.write("Use the sidebar to input your information and get charges predictions.")
