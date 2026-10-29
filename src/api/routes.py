"""HTTP routes: take the request, call the service, return the response."""
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from api.schemas import PredictRequest, PredictResponse
from api.service import PredictionService

router = APIRouter()

def get_service(request: Request) -> PredictionService:
    return request.app.state.service

ServiceDep = Annotated[PredictionService, Depends(get_service)]

@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, service: ServiceDep) -> PredictResponse:
    return service.predict(body.title, body.description)

@router.get("/health")
def health(service: ServiceDep) -> dict:
    return {"status": "ok", "models": service.versions()}