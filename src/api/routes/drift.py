# ============================================================
# FILE: src/api/routes/drift.py
# ============================================================
"""API endpoints for drift monitoring."""
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.monitoring.drift_detector import check_drift_and_alert

router = APIRouter()


class DriftStatus(BaseModel):
    """Drift detection status."""
    timestamp: str
    alert: bool
    input_drift_detected: bool
    prediction_drift_detected: bool
    message: str


@router.get("/drift/status", response_model=DriftStatus)
async def get_drift_status():
    """
    Get current drift detection status.

    Compares recent predictions (last 24h) against reference window (last 7 days).
    """
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'ecg_user')}:" \
             f"{os.getenv('POSTGRES_PASSWORD', 'ecg_password_dev')}@" \
             f"{os.getenv('POSTGRES_HOST', 'localhost')}:" \
             f"{os.getenv('POSTGRES_PORT', '5432')}/" \
             f"{os.getenv('POSTGRES_DB', 'ecg_predictions')}"

    try:
        summary = await check_drift_and_alert(db_url, threshold=0.1)

        return DriftStatus(
            timestamp=summary['timestamp'],
            alert=summary['alert'],
            input_drift_detected=summary['input_drift'].get('drift_detected', False),
            prediction_drift_detected=summary['prediction_drift'].get('drift_detected', False),
            message="Drift detected - investigate immediately" if summary['alert']
                    else "No drift detected - model performing normally"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Drift detection failed: {str(e)}"
        )


@router.get("/drift/metrics")
async def get_drift_metrics():
    """Get detailed drift metrics for monitoring dashboard."""
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'ecg_user')}:" \
             f"{os.getenv('POSTGRES_PASSWORD', 'ecg_password_dev')}@" \
             f"{os.getenv('POSTGRES_HOST', 'localhost')}:" \
             f"{os.getenv('POSTGRES_PORT', '5432')}/" \
             f"{os.getenv('POSTGRES_DB', 'ecg_predictions')}"

    try:
        summary = await check_drift_and_alert(db_url, threshold=0.1)
        return summary
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get drift metrics: {str(e)}"
        )
