from typing import List
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path

from api.schemas import (
    InsuranceChargePredictRequest, 
    InsuranceChargePredictResponse, 
    BatchInsuranceChargePredictRequest
)

# Mount the model directory
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "trained"

# Load the model and preprocessor
try:
    model = joblib.load(MODEL_DIR / "insurance_charges_model_xgbregressor.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor_ordinal_clean.pkl")

except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    model = None
    preprocessor = None

def predict_insurance_charge(request: InsuranceChargePredictRequest) -> InsuranceChargePredictResponse:
    """
    Predict health insurance charges based on input features.
    Args:
        request (InsuranceChargePredictRequest): Input features for prediction.
    Returns:

        InsuranceChargePredictResponse: Predicted insurance charge.
    """
    if model is None or preprocessor is None:
        raise RuntimeError("Model or preprocessor not loaded properly.")

    # Convert request to DataFrame (from json to dict to DataFrame)
    input_data = pd.DataFrame([request.model_dump()])
    
    # Preprocess the input data
    processed_data = preprocessor.transform(input_data)
    
    # Make prediction
    predicted_charge = model.predict(processed_data)[0]

    # Round the prediction to 2 decimal places
    predicted_charge = round(float(predicted_charge), 2)

    return InsuranceChargePredictResponse(
        predicted_charge=predicted_charge,
        prediction_time=datetime.now().isoformat()
    )

def batch_predict_insurance_charges(batch_request: BatchInsuranceChargePredictRequest) -> List[InsuranceChargePredictResponse]:
    """
    Predict health insurance charges for a batch of requests.
    Args:
        batch_request (BatchInsuranceChargePredictRequest): Batch of input features for prediction.
    Returns:
        List[InsuranceChargePredictResponse]: List of predicted insurance charges.
    """
    if model is None or preprocessor is None:
        raise RuntimeError("Model or preprocessor not loaded properly.")
    
    # Convert batch request to DataFrame
    input_data = pd.DataFrame([req.model_dump() for req in batch_request.requests])

    # Preprocess the input data
    processed_data = preprocessor.transform(input_data)

    # Make predictions
    predicted_insurance_charges = model.predict(processed_data)

    # Round the predictions to 2 decimal places and create response objects
    prediction_time = datetime.now().isoformat()
    return [
        InsuranceChargePredictResponse(
            predicted_charge=round(float(charge), 2),
            prediction_time=prediction_time
        )
        for charge in predicted_insurance_charges
    ]
