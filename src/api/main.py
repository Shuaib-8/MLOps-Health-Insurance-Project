from fastapi import FastAPI

# Initialize the FastAPI app with metadata
app = FastAPI(
    title="Health Insurance US Charges Prediction API",
    description="""API for predicting health insurance US charges using a trained machine learning model.""",
    version="1.0.0",
)


# Health check endpoint
@app.get("/health", response_model=dict)
async def health_check():
    """
    Health check endpoint to verify that the API is running.
    """
    return {"status": "ok", "message": "API is running"}
