from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api.inference import predict_insurance_charge, batch_predict_insurance_charges
from api.schemas import InsuranceChargePredictRequest, InsuranceChargePredictResponse, BatchInsuranceChargePredictRequest

# Initialize the FastAPI app with metadata
app = FastAPI(
    title="Health Insurance US Charges Prediction API",
    description="""API for predicting health insurance US charges using a trained machine learning model.""",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=dict)
@app.head("/health", response_model=dict)
async def health_check():
    """
    Health check endpoint to verify that the API is running.
    """
    return {"status": "ok", "message": "API is running"}

# Import the prediction endpoint from the inference module
@app.post("/predict", response_model=dict)
async def predict(request: dict):
    """
    Endpoint to predict health insurance charges.
    """
    try:
        # Parse the incoming request to the Pydantic model
        request_data = InsuranceChargePredictRequest(**request)

        # Get prediction
        response = predict_insurance_charge(request_data)
        return {"predicted_charge": response.predicted_charge,
                "prediction_time": response.prediction_time}
    except Exception as e:
        return {"error": str(e)}


# Batch prediction endpoint
@app.post("/batch-predict", response_model=list)
async def batch_predict_endpoint(requests: list[InsuranceChargePredictRequest]):
    """
    Endpoint to predict health insurance charges for multiple requests.
    """
    try:
        # Wrap the list in BatchInsuranceChargePredictRequest
        batch_request = BatchInsuranceChargePredictRequest(requests=requests)

        # Get batch predictions
        responses = batch_predict_insurance_charges(batch_request)
        return [{"predicted_charge": resp.predicted_charge, "prediction_time": resp.prediction_time}
                for resp in responses]
    except Exception as e:
        return {"error": str(e)}


# Entry point function for package installation
def main():
    """Entry point for the API server when installed as package."""
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
