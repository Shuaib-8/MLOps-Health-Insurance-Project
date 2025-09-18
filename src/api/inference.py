import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path

from api.schemas import InsuranceChargePredictRequest, InsuranceChargePredictResponse

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
    if model is None or preprocessor is None:
        raise RuntimeError("Model or preprocessor not loaded properly.")
    
    # Convert request to DataFrame
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
