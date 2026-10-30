"""FastAPI service: role and level of a job posting from its title and description."""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.registry import load_production
from api.routes import router
from api.service import PredictionService
from models.constants import TARGETS
from models.tracking import setup_mlflow

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_mlflow()
    app.state.service = PredictionService({target: load_production(target) for target in TARGETS})
    yield

app = FastAPI(title="Vacancy role classifier", lifespan=lifespan)
app.include_router(router)