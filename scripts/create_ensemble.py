#!/usr/bin/env python3
# ============================================================
# FILE: scripts/create_ensemble.py
# ============================================================
"""
Create ensemble model by averaging predictions from multiple models.

Ensemble often improves performance without additional training.
Combines: ResNet-18 baseline + Focal Loss model (+ optional ResNet-34)

Usage:
    PYTHONPATH=. python scripts/create_ensemble.py \
        --models models/best_model.pth models/best_model_focal.pth
"""
import argparse
import ast
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.ml.models.resnet1d import resnet18_1d, resnet34_1d

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
# Ensemble Predictor
# ============================================================

class EnsemblePredictor:
    """Ensemble of multiple models."""

    def __init__(self, models, device):
        self.models = models
        self.device = device

        for model in self.models:
            model.eval()

    def predict(self, signal):
        """
        Average predictions from all models.

        Args:
            signal: ECG signal tensor (batch, 12, 5000)

        Returns:
            Averaged probabilities (batch, num_classes)
        """
        predictions = []

        with torch.no_grad():
            for model in self.models:
                outputs = model(signal)
                probs = torch.sigmoid(outputs)
                predictions.append(probs)

        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)

        return ensemble_pred


def evaluate_ensemble(ensemble, loader, device):
    """Evaluate ensemble model."""
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Ensemble Evaluation"):
        signals = batch['signal'].to(device)
        labels = batch['labels'].to(device)

        preds = ensemble.predict(signals)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Calculate AUC per class
    aucs = []
    per_class_aucs = {}

    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)
            per_class_aucs[i] = auc

    macro_auc = np.mean(aucs) if aucs else 0.0

    return macro_auc, per_class_aucs, all_preds, all_labels


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Paths to model files to ensemble'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
    )
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-path', type=str, default='models/ensemble_weights.json')

    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("🤝 Ensemble Model Creation")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Models to ensemble: {len(args.models)}")
    for i, model_path in enumerate(args.models, 1):
        print(f"   {i}. {model_path}")
    print()

    # Load data
    data_dir = Path(args.data_dir)
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

    # Load all models
    print("📦 Loading models...")
    models = []

    for model_path_str in args.models:
        model_path = Path(model_path_str)

        # Detect architecture from filename
        if 'resnet34' in model_path.stem:
            model = resnet34_1d(num_classes=len(SUPERCLASSES))
        else:
            model = resnet18_1d(num_classes=len(SUPERCLASSES))

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)

        models.append(model)
        print(f"   ✅ Loaded: {model_path.name}")

    # Create ensemble
    print(f"\n🤝 Creating ensemble of {len(models)} models...")
    ensemble = EnsemblePredictor(models, device)

    # Evaluate ensemble
    print("🔮 Evaluating ensemble...")
    ensemble_auc, per_class_aucs, ensemble_preds, ensemble_labels = evaluate_ensemble(
        ensemble, test_loader, device
    )

    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS")
    print("=" * 60)

    print(f"\nEnsemble Test AUC: {ensemble_auc:.4f}")

    print("\nPer-class AUC:")
    for i, sc in enumerate(SUPERCLASSES):
        if i in per_class_aucs:
            auc = per_class_aucs[i]
            print(f"  {sc}: {auc:.4f}")

    # Compare to baseline (ResNet-18 v1)
    baseline_aucs = {
        'NORM': 0.9610,
        'MI': 0.9422,
        'STTC': 0.9320,
        'CD': 0.9461,
        'HYP': 0.8775,
    }
    baseline_macro = np.mean(list(baseline_aucs.values()))

    print("\nComparison to Baseline (ResNet-18 v1):")
    print(f"  Baseline AUC:  {baseline_macro:.4f}")
    print(f"  Ensemble AUC:  {ensemble_auc:.4f}")
    improvement = ensemble_auc - baseline_macro
    print(f"  Improvement:   {improvement:+.4f} ({improvement/baseline_macro:+.2%})")

    # Per-class comparison
    print("\nPer-class Improvements:")
    for i, sc in enumerate(SUPERCLASSES):
        if i in per_class_aucs:
            ensemble_auc_class = per_class_aucs[i]
            baseline_auc_class = baseline_aucs[sc]
            diff = ensemble_auc_class - baseline_auc_class

            status = "✅" if diff > 0.01 else "→" if abs(diff) < 0.01 else "⚠️"
            print(f"  {sc}: {baseline_auc_class:.4f} → {ensemble_auc_class:.4f} ({diff:+.4f}) {status}")

    # Log to MLflow
    mlflow.set_experiment("ensemble-models")

    with mlflow.start_run(run_name="ensemble_" + "_".join([Path(m).stem for m in args.models[:2]])):
        mlflow.log_params({
            "num_models": len(args.models),
            "model_paths": str(args.models),
            "ensemble_method": "simple_average",
        })

        mlflow.log_metrics({
            "test_auc": ensemble_auc,
            "improvement_over_baseline": improvement,
            **{f"test_auc_{sc}": per_class_aucs[i] for i, sc in enumerate(SUPERCLASSES) if i in per_class_aucs}
        })

        print("\n📊 Ensemble logged to MLflow")
        print("   Experiment: ensemble-models")

    print("\n" + "=" * 60)
    print("✅ Ensemble analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
