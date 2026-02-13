#!/usr/bin/env python3
# ============================================================
# FILE: scripts/evaluate_calibration.py
# ============================================================
"""
Calibration analysis for clinical deployment readiness.

Evaluates if model confidence scores match actual accuracy.
Critical for healthcare: "90% confidence" should mean 90% correct.

Usage:
    PYTHONPATH=. python scripts/evaluate_calibration.py \
        --model-path models/best_model.pth
"""
import argparse
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.ml.models.resnet1d import resnet18_1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Dataset
# ============================================================

class PTBXLDataset(Dataset):
    """PTB-XL dataset."""

    def __init__(self, df: pd.DataFrame, records_dir: Path):
        self.df = df.reset_index(drop=True)
        self.records_dir = Path(records_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        record_path = self.records_dir.parent / row.filename_hr
        record = wfdb.rdsamp(str(record_path.with_suffix('')))
        signal = record[0].T.astype(np.float32)

        signal = (signal - signal.mean(axis=1, keepdims=True)) / (
            signal.std(axis=1, keepdims=True) + 1e-8
        )

        labels = np.array(row.labels, dtype=np.float32)

        return {
            'signal': torch.from_numpy(signal),
            'labels': torch.from_numpy(labels),
        }


# ============================================================
# Calibration Functions
# ============================================================

def calculate_ece(y_true, y_pred, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures the difference between confidence and accuracy.
    Lower is better. <0.05 is considered well-calibrated.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy='uniform'
    )

    # Weight by number of samples in each bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_weights = bin_counts / len(y_pred)

    # Weighted average of absolute differences
    ece = np.sum(bin_weights[:len(fraction_of_positives)] *
                 np.abs(fraction_of_positives - mean_predicted_value))

    return ece, fraction_of_positives, mean_predicted_value


def plot_calibration_curve(y_true, y_pred, diagnosis_name, output_path):
    """Generate calibration curve plot."""

    ece, frac_pos, mean_pred = calculate_ece(y_true, y_pred, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

    # Model calibration
    ax.plot(mean_pred, frac_pos, 'o-', linewidth=2, markersize=8,
            label=f'{diagnosis_name} (ECE: {ece:.3f})')

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Probability', fontsize=12)
    ax.set_title(f'Calibration Curve - {diagnosis_name}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Add ECE annotation
    status = "✅ Well-calibrated" if ece < 0.05 else "⚠️ Needs calibration"
    ax.text(0.05, 0.95, f'ECE: {ece:.4f}\n{status}',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top',
            fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return ece


# ============================================================
# Model Evaluation
# ============================================================

def evaluate_model(model, loader, device):
    """Get predictions for calibration analysis."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            signals = batch['signal'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(signals)
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    return all_preds, all_labels


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    )
    parser.add_argument('--output-dir', type=str, default='reports/calibration')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()

    # Setup
    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("📊 Calibration Analysis")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print()

    # Load data
    print("📁 Loading test data...")
    database_file = data_dir / "ptbxl_database.csv"
    records_dir = data_dir / "records500"

    df = pd.read_csv(database_file)
    df['scp_codes_dict'] = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Label processing
    SUPERCLASS_MAP = {
        'NORM': 'NORM',
        'IMI': 'MI', 'AMI': 'MI', 'LMI': 'MI', 'ALMI': 'MI', 'ILMI': 'MI',
        'ASMI': 'MI', 'PMI': 'MI', 'INJAL': 'MI', 'IPLMI': 'MI', 'IPMI': 'MI',
        'NST_': 'STTC', 'DIG': 'STTC', 'LNGQT': 'STTC', 'ISC_': 'STTC',
        'STTC': 'STTC', 'STD_': 'STTC', 'STE_': 'STTC',
        'IRBBB': 'CD', 'CRBBB': 'CD', 'CLBBB': 'CD', 'ILBBB': 'CD',
        'LAFB': 'CD', 'LPFB': 'CD', 'IVCD': 'CD', 'WPW': 'CD',
        'LVH': 'HYP', 'RVH': 'HYP', 'LAO/LAE': 'HYP', 'RAO/RAE': 'HYP',
    }
    SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

    def get_superclass_labels(scp_dict, mapping):
        labels = set()
        for code in scp_dict.keys():
            if code in mapping:
                labels.add(mapping[code])
        return labels

    def encode_labels(label_set):
        return [1 if sc in label_set else 0 for sc in SUPERCLASSES]

    df['superclass_labels'] = df.scp_codes_dict.apply(
        lambda x: get_superclass_labels(x, SUPERCLASS_MAP)
    )
    df = df[df.superclass_labels.apply(len) > 0].reset_index(drop=True)
    df['labels'] = df.superclass_labels.apply(encode_labels)

    # Test split
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)

    test_dataset = PTBXLDataset(test_df, records_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # Load model
    print("📦 Loading model...")
    model = resnet18_1d(num_classes=len(SUPERCLASSES))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("✅ Model loaded\n")

    # Get predictions
    print("🔮 Generating predictions...")
    predictions, labels = evaluate_model(model, test_loader, device)

    # Analyze calibration for each class
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)

    calibration_results = {}

    for i, diagnosis in enumerate(SUPERCLASSES):
        print(f"\n📊 {diagnosis}:")

        y_true = labels[:, i]
        y_pred = predictions[:, i]

        # Calculate ECE
        ece, frac_pos, mean_pred = calculate_ece(y_true, y_pred, n_bins=10)

        # Generate plot
        plot_path = output_dir / f"calibration_{diagnosis}.png"
        plot_calibration_curve(y_true, y_pred, diagnosis, plot_path)

        status = "✅ Well-calibrated" if ece < 0.05 else "⚠️ Needs recalibration"
        print(f"   ECE: {ece:.4f} {status}")
        print(f"   Plot saved: {plot_path}")

        calibration_results[diagnosis] = {
            'ece': float(ece),
            'well_calibrated': ece < 0.05,
        }

    # Overall summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    avg_ece = np.mean([r['ece'] for r in calibration_results.values()])
    well_calibrated_count = sum([r['well_calibrated'] for r in calibration_results.values()])

    print(f"Average ECE: {avg_ece:.4f}")
    print(f"Well-calibrated classes: {well_calibrated_count}/{len(SUPERCLASSES)}")

    if avg_ece < 0.05:
        print("\n✅ Model is well-calibrated for clinical deployment!")
    elif avg_ece < 0.10:
        print("\n⚠️  Model needs minor calibration (consider temperature scaling)")
    else:
        print("\n❌ Model needs significant calibration before deployment")

    # Log to MLflow
    mlflow.set_experiment("calibration-analysis")

    with mlflow.start_run(run_name=f"calibration_{model_path.stem}"):
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_metric("avg_ece", avg_ece)
        mlflow.log_metric("well_calibrated_count", well_calibrated_count)

        for diagnosis, results in calibration_results.items():
            mlflow.log_metric(f"ece_{diagnosis}", results['ece'])
            mlflow.log_artifact(str(output_dir / f"calibration_{diagnosis}.png"))

        print("\n📊 Results logged to MLflow")
        print("   View: http://localhost:5000")


if __name__ == "__main__":
    main()
