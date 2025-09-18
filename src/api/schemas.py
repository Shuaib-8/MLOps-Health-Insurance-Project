# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional


class InsuranceChargePredictRequest(BaseModel):
    age: int = Field(..., ge=18, le=64, description="Age of the individual")
    bmi: float = Field(
        ..., ge=15.96, le=53.13, description="Body Mass Index (BMI) of the individual"
    )
    children: int = Field(
        ...,
        ge=0,
        le=5,
        description="Number of children/dependents covered by the insurance",
    )
    sex: str = Field(..., description="Gender of the individual (male/female)")
    smoker: str = Field(
        ..., description="Whether the individual is a smoker or not (yes/no)"
    )
    region: str = Field(
        ...,
        description="Region where the individual resides (northeast, northwest, southeast, southwest)",
    )


class InsuranceChargePredictResponse(BaseModel):
    predicted_charge: float = Field(
        ..., description="Predicted health insurance charge for the individual"
    )
    prediction_time: str = Field(
        ..., description="ISO format timestamp when prediction was made"
    )
