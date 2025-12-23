# ============================================================
# FILE: src/api/routes/health.py
# ============================================================
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Liveness probe - is the service running?"""
    return {"status": "healthy"}


@router.get("/ready")
async def readiness_check():
    """Readiness probe - is the service ready to accept traffic?"""
    # TODO: Check DB connection, model loaded, etc.
    return {"status": "ready", "checks": {"database": "ok", "model": "ok"}}
