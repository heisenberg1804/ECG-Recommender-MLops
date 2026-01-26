# ============================================================
# FILE: src/api/routes/bias.py
# ============================================================
"""API endpoints for bias monitoring."""
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.monitoring.bias_analyzer import check_bias_and_alert

router = APIRouter()


class BiasStatus(BaseModel):
    """Bias monitoring status."""
    timestamp: str
    bias_detected: bool
    sample_size: int
    sex_parity_passes: bool
    age_parity_passes: bool
    warnings: list[str]
    summary: str


@router.get("/bias/status", response_model=BiasStatus)
async def get_bias_status():
    """
    Get current bias monitoring status.
    
    Analyzes last 7 days of predictions for fairness across demographics.
    """
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'ecg_user')}:" \
             f"{os.getenv('POSTGRES_PASSWORD', 'ecg_password_dev')}@" \
             f"{os.getenv('POSTGRES_HOST', 'localhost')}:" \
             f"{os.getenv('POSTGRES_PORT', '5432')}/" \
             f"{os.getenv('POSTGRES_DB', 'ecg_predictions')}"

    try:
        report = await check_bias_and_alert(db_url)

        sex_passes = report['demographic_parity'].get('sex_parity', {}).get('passes', True)
        age_passes = report['demographic_parity'].get('age_parity', {}).get('passes', True)

        return BiasStatus(
            timestamp=report['timestamp'],
            bias_detected=report['bias_detected'],
            sample_size=report.get('sample_size', 0),
            sex_parity_passes=sex_passes,
            age_parity_passes=age_passes,
            warnings=report['warnings'],
            summary=report['summary']
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Bias analysis failed: {str(e)}"
        )


@router.get("/bias/metrics")
async def get_bias_metrics() -> dict[str, Any]:
    """Get detailed bias metrics for dashboard."""
    import numpy as np

    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'ecg_user')}:" \
             f"{os.getenv('POSTGRES_PASSWORD', 'ecg_password_dev')}@" \
             f"{os.getenv('POSTGRES_HOST', 'localhost')}:" \
             f"{os.getenv('POSTGRES_PORT', '5432')}/" \
             f"{os.getenv('POSTGRES_DB', 'ecg_predictions')}"

    try:
        report = await check_bias_and_alert(db_url)

        # Convert numpy types to Python native types
        def convert_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj

        return convert_types(report)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get bias metrics: {str(e)}"
        )
