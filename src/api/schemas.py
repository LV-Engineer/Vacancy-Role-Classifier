"""Request and response schemas of the service."""
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300, examples=["Senior Data Analyst"])
    description: str = Field(default="", max_length=20000)

class Prediction(BaseModel):
    label: str
    probability: float
    probabilities: dict[str, float]

class PredictResponse(BaseModel):
    role: Prediction
    level: Prediction