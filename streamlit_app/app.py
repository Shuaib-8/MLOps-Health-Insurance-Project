import datetime
import os
from unittest import result 

import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="Health Insurance Charge Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# App title and description
st.markdown('<h1 class="main-header">Health Insurance Charge Prediction</h1>', unsafe_allow_html=True)
st.markdown("""
            <p style="font-size:18px; color:gray;">
            This application predicts US health insurance Charges (annual) based on various factors. 
            Inference is performed using a trained XGBoost model. 
            </p>
            """,
            unsafe_allow_html=True
)

# Create two columns for layout
col1, col2 = st.columns(2, gap="large")

# Initialize session state for prediction results
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'prediction_error' not in st.session_state:
    st.session_state.prediction_error = None

# Input form in the first (main) column
with col1:
    st.subheader("Input Features")

    # BMI input
    bmi = st.slider(
        "BMI",
        min_value=15.96,
        max_value=53.13,
        value=25.0,
        step=0.1,
        format="%.2f",
        help="Body Mass Index (15.96-53.13)"
    )

    # Age input
    age = st.slider(
        "Age",
        min_value=18,
        max_value=64,
        value=30,
        step=1,
        help="Age of the individual (18-64 years)"
    )

    # Children input
    children = st.selectbox(
        "Number of Children",
        options=[0, 1, 2, 3, 4, 5],
        index=0,
        help="Number of children/dependents covered by insurance"
    )

    sex_col, smoker_col, region_col = st.columns(3)

    with sex_col:
        # Sex input
        sex = st.selectbox(
            "Sex",
            options=["female", "male"],
        index=0,
        help="Gender of the individual"
    )

    with smoker_col:
        # Smoker input
        smoker = st.selectbox(
            "Smoker",
            options=["no", "yes"],
            index=0,
            help="Whether the individual is a smoker"
        )

    with region_col:
        # Region input
        region = st.selectbox(
            "Region",
            options=["northeast", "northwest", "southeast", "southwest"],
            index=0,
        help="Region where the individual resides"
    )

    # Predict button
    predict_button = st.button("Predict Charges", type="primary")

# Prediction results in the second column
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2>Prediction Result</h2>', unsafe_allow_html=True)

    # Handle prediction logic
    if predict_button:
        # Show loading spinner
        with st.spinner('Predicting insurance charges...'):
            try:
                # Prepare request data
                request_data = {
                    "age": age,
                    "bmi": bmi,
                    "children": children,
                    "sex": sex,
                    "smoker": smoker,
                    "region": region
                }

                # Make API request
                API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
                predict_url = f"{API_ENDPOINT}/predict"

                st.write(f"Sending request to API at: {predict_url}")  # Debug line


                response = requests.post(
                    predict_url,
                    json=request_data,
                    timeout=10
                )
                response.raise_for_status()  # Raise an error for bad status codes
                prediction = response.json()

                if response.status_code == 200:
                    prediction = response.json()
                    st.session_state.prediction_result = prediction
                    st.session_state.prediction_error = None
                else:
                    st.session_state.prediction_error = f"API request failed with status code: {response.status_code}"
                    st.session_state.prediction_result = None

            except requests.exceptions.ConnectionError:
                st.session_state.prediction_error = "Could not connect to the prediction API. Please ensure the FastAPI server is running on localhost:8000."
                st.session_state.prediction_result = None
            except requests.exceptions.Timeout:
                st.session_state.prediction_error = "Request timed out. Please try again."
                st.session_state.prediction_result = None
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred connecting to the API: {e}")
                st.warning("Using mock prediction for demonstration purposes. Please check your API connection.")
                # Mock prediction for demonstration
                mock_prediction = {
                    "predicted_charge": 12345.67,
                    "prediction_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.prediction_result = mock_prediction
                st.session_state.prediction_error = None    


    # Display results or placeholder
    if "prediction_result" in st.session_state and st.session_state.prediction_result is not None:
        if st.session_state.prediction_error:
            st.error(st.session_state.prediction_error)
        else:
            predicted_charge = st.session_state.prediction_result.get('predicted_charge', 0)
            prediction_time = st.session_state.prediction_result.get('prediction_time', '')

            # Formatting the result display
            predicted_charge = f"${predicted_charge:,.2f}"
            prediction_time = f"Prediction made at (UTC Time zone): \n\n {prediction_time}"

            # Display prediction charge and time
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-label">Predicted Annual Insurance Charge</div>
                    <div class="prediction-amount">{predicted_charge}</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="prediction-time">{prediction_time}</div>
                """, unsafe_allow_html=True)
    elif "prediction_result" in st.session_state and st.session_state.prediction_error:
        st.error(st.session_state.prediction_error)
    else:
        # Show placeholder text when no prediction has been made
        st.markdown("""    
        <div style="text-align: center; padding: 40px; color: #666; font-size: 16px;">
            Fill out the form and click "Predict Charges" to see the estimated insurance charge.
        </div>
        """, unsafe_allow_html=True)
