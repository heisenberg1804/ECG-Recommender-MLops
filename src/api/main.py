# ============================================================
# FILE: src/api/main.py (UPDATED WITH PROMETHEUS)
# ============================================================
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.routes import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model, connect to DB, etc.
    print("Starting up...")
    yield
    # Shutdown: Clean up resources
    print("Shutting down...")


app = FastAPI(
    title="ECG Clinical Action Recommender",
    description="Recommends clinical actions based on 12-lead ECG signals",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Predictions"])
