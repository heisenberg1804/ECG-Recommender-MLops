# ============================================================
# FILE: src/api/routes/predict.py (UPDATED WITH METRICS)
# ============================================================
import time
import uuid

import numpy as np
from fastapi import APIRouter, HTTPException
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field

from src.ml.inference.predictor import get_predictor

router = APIRouter()

# Prometheus metrics
PREDICTIONS_TOTAL = Counter(
    'ecg_predictions_total',
    'Total number of predictions made',
    ['diagnosis', 'urgency']
)

PREDICTION_LATENCY = Histogram(
    'ecg_prediction_latency_seconds',
    'Time spent processing predictions',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

PREDICTION_CONFIDENCE = Histogram(
    'ecg_prediction_confidence',
    'Confidence scores of predictions',
    ['diagnosis'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

MODEL_LOADED = Gauge(
    'ecg_model_loaded',
    'Whether the model is loaded (1) or not (0)'
)


class ECGInput(BaseModel):
    """Input schema for ECG prediction."""

    ecg_signal: list[list[float]] = Field(
        ...,
        description="12-lead ECG signal, shape (12, num_samples)",
        min_length=12,
        max_length=12,
    )
    patient_age: int | None = Field(None, ge=0, le=120)
    patient_sex: str | None = Field(None, pattern="^(M|F)$")
    sampling_rate: int = Field(default=500, description="Sampling rate in Hz")


class ClinicalAction(BaseModel):
    """A recommended clinical action."""

    action: str
    confidence: float = Field(ge=0.0, le=1.0)
    urgency: str = Field(pattern="^(immediate|urgent|routine)$")
    reasoning: str


class Diagnosis(BaseModel):
    """A predicted diagnosis."""

    diagnosis: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Response schema for ECG prediction."""

    ecg_id: str
    diagnoses: list[Diagnosis]
    recommendations: list[ClinicalAction]
    model_version: str
    processing_time_ms: float


@router.post("/predict", response_model=PredictionResponse)
async def predict_clinical_actions(ecg_input: ECGInput) -> PredictionResponse:
    """
    Predict clinical actions based on ECG signal.

    This endpoint accepts a 12-lead ECG signal and returns
    recommended clinical actions ranked by confidence.

    The model classifies the ECG into diagnostic categories:
    - NORM: Normal ECG
    - MI: Myocardial Infarction
    - STTC: ST/T Changes
    - CD: Conduction Disturbance
    - HYP: Hypertrophy

    Then maps diagnoses to evidence-based clinical actions.
    """
    start_time = time.time()

    # Validate signal shape
    signal = np.array(ecg_input.ecg_signal, dtype=np.float32)
    if signal.shape[0] != 12:
        raise HTTPException(
            status_code=400,
            detail=f"ECG must have exactly 12 leads, got {signal.shape[0]}"
        )

    try:
        # Get predictor (lazy loads model on first call)
        predictor = get_predictor()
        MODEL_LOADED.set(1)

        # Run inference (track latency)
        with PREDICTION_LATENCY.time():
            result = predictor.predict(signal, threshold=0.3, top_k=5)

        # Convert to response format
        diagnoses = [
            Diagnosis(diagnosis=d['diagnosis'], confidence=d['confidence'])
            for d in result['diagnoses']
        ]

        recommendations = [
            ClinicalAction(
                action=r['action'],
                confidence=r['confidence'],
                urgency=r['urgency'],
                reasoning=r['reasoning'],
            )
            for r in result['recommendations']
        ]

        # Record metrics
        for dx in diagnoses:
            PREDICTION_CONFIDENCE.labels(diagnosis=dx.diagnosis).observe(dx.confidence)

        for rec in recommendations:
            # Infer diagnosis from recommendations
            diagnosis = next((d.diagnosis for d in diagnoses), 'UNKNOWN')
            PREDICTIONS_TOTAL.labels(
                diagnosis=diagnosis,
                urgency=rec.urgency
            ).inc()

    except Exception as e:
        MODEL_LOADED.set(0)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    processing_time = (time.time() - start_time) * 1000

    return PredictionResponse(
        ecg_id=str(uuid.uuid4()),
        diagnoses=diagnoses,
        recommendations=recommendations,
        model_version="resnet18-v0.1.0",
        processing_time_ms=round(processing_time, 2),
    )


@router.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        predictor = get_predictor()
        MODEL_LOADED.set(1)
        return {
            "model_type": "ResNet-18 1D",
            "classes": predictor.superclasses,
            "device": str(predictor.device),
            "model_path": str(predictor.model_path),
        }
    except Exception as e:
        MODEL_LOADED.set(0)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model info: {str(e)}"
        )
