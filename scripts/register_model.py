#!/usr/bin/env python3
# ============================================================
# FILE: scripts/register_model.py
# ============================================================
"""
Register trained model in MLflow Model Registry.

This allows the model to be downloaded by the API at runtime
instead of being baked into the Docker image.

Usage:
    python scripts/register_model.py --run-id <mlflow_run_id>
"""
import argparse
from pathlib import Path

import mlflow


def register_model(run_id: str, model_name: str = "ecg-clinical-recommender"):
    """
    Register a model from an MLflow run.

    Args:
        run_id: MLflow run ID containing the model
        model_name: Name to register the model under
    """
    print(f"📦 Registering model from run: {run_id}")

    # Set tracking URI to local DB
    mlflow_db = Path(__file__).parent.parent / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")

    # Register model
    model_uri = f"runs:/{run_id}/model"

    try:
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        print("✅ Model registered!")
        print(f"   Name: {model_name}")
        print(f"   Version: {model_version.version}")
        print(f"   Run ID: {run_id}")

        # Transition to Production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production"
        )

        print("✅ Model transitioned to Production stage")

        return model_version

    except Exception as e:
        print(f"❌ Failed to register model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="MLflow run ID containing the trained model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ecg-clinical-recommender",
        help="Name to register model under"
    )

    args = parser.parse_args()

    register_model(args.run_id, args.model_name)

    print("\n🎉 Done! Model is now available in MLflow Model Registry")
    print("\nNext steps:")
    print("  1. Update API to load from registry")
    print("  2. Rebuild Docker image")
    print("  3. Deploy to K8s")


if __name__ == "__main__":
    main()
