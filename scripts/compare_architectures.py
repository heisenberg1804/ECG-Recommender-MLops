"""
Train multiple architectures and compare in MLflow.

Builds on existing train.py but adds architecture variations.
"""
import sys
from pathlib import Path

import mlflow

from src.ml.models.resnet1d import resnet18_1d, resnet34_1d

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


ARCHITECTURES = {
    'resnet18': {
        'model_fn': lambda: resnet18_1d(num_classes=5),
        'batch_size': 64,
        'epochs': 30,
    },
    'resnet34': {
        'model_fn': lambda: resnet34_1d(num_classes=5),
        'batch_size': 64,
        'epochs': 30,
    },
    # Add more architectures as you implement them
}

def train_architecture(arch_name, config):
    """Train single architecture and log to MLflow."""

    # Set experiment (creates if doesn't exist)
    mlflow.set_experiment("architecture_comparison")

    with mlflow.start_run(run_name=arch_name):
        # Log architecture metadata
        mlflow.log_param("architecture", arch_name)
        mlflow.log_param("data_version", "ptb-xl-v1.0")
        mlflow.log_param("sampling_ratio", 1.0)

        # Import and use existing training logic from train.py
        from scripts.train import main as train_main

        # Train (reuse existing train.py logic)
        metrics = train_main(
            model=config['model_fn'](),
            batch_size=config['batch_size'],
            epochs=config['epochs']
        )

        # Metrics already logged in train.py
        return metrics

if __name__ == "__main__":
    for arch_name, config in ARCHITECTURES.items():
        print(f"\n{'='*60}")
        print(f"Training: {arch_name}")
        print(f"{'='*60}\n")

        train_architecture(arch_name, config)

    print("\n✅ All architectures trained!")
    print("View comparison: mlflow ui --port 5000")
