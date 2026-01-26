#!/usr/bin/env python3
# ============================================================
# FILE: scripts/train_v2.py
# ============================================================
"""
Train improved ResNet-18 v2 on PTB-XL dataset.

Improvements over v1:
- Early stopping (save training time)
- Class weighting (improve HYP performance)
- Dropout regularization (reduce overfitting)
- Auto-registration to MLflow Model Registry

Usage:
    PYTHONPATH=. python scripts/train_v2.py
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
import torch.nn as nn
import torch.optim as optim
import wfdb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.ml.models.resnet1d import resnet18_1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    """Early stopping to stop training when validation doesn't improve."""

    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'max' and score >= self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0


# ============================================================
# Dataset
# ============================================================

class PTBXLDataset(Dataset):
    """PTB-XL dataset for ECG signals."""

    def __init__(self, df: pd.DataFrame, records_dir: Path):
        self.df = df.reset_index(drop=True)
        self.records_dir = Path(records_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load ECG signal
        record_path = self.records_dir.parent / row.filename_hr
        record = wfdb.rdsamp(str(record_path.with_suffix('')))
        signal = record[0].T.astype(np.float32)

        # Normalize
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (
            signal.std(axis=1, keepdims=True) + 1e-8
        )

        labels = np.array(row.labels, dtype=np.float32)

        return {
            'signal': torch.from_numpy(signal),
            'labels': torch.from_numpy(labels),
        }


# ============================================================
# Training Functions
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        signals = batch['signal'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            signals = batch['signal'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    avg_loss = total_loss / len(loader)

    # Calculate AUC per class
    aucs = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)

    macro_auc = np.mean(aucs) if aucs else 0.0

    return avg_loss, macro_auc, all_preds, all_labels


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                       default='../data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
    parser.add_argument('--output-dir', type=str, default='models')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--auto-register', action='store_true',
                         help='Auto-register to MLflow if AUC > 0.90')

    args = parser.parse_args()

    # Setup
    data_dir = Path(args.data_dir)
    records_dir = data_dir / "records500"
    database_file = data_dir / "ptbxl_database.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("🖥️  Model v2 Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Dropout: {args.dropout}")
    print(f"Early stopping patience: {args.patience}")
    print()

    # Load data
    print("📁 Loading data...")
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

    print(f"   Records: {len(df)}")
    for i, sc in enumerate(SUPERCLASSES):
        count = sum(df.labels.apply(lambda x: x[i]))
        print(f"   {sc}: {count} ({count/len(df)*100:.1f}%)")

    # Compute class weights for HYP improvement
    print("\n⚖️  Computing class weights...")
    class_weights = []
    for i in range(len(SUPERCLASSES)):
        class_labels = df.labels.apply(lambda x: x[i]).values
        if len(np.unique(class_labels)) > 1:
            weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=class_labels)
            pos_weight = weights[1] / weights[0]
            class_weights.append(pos_weight)
        else:
            class_weights.append(1.0)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"   Class weights: {class_weights}")

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    print(f"\n🔀 Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Datasets
    train_dataset = PTBXLDataset(train_df, records_dir)
    val_dataset = PTBXLDataset(val_df, records_dir)
    test_dataset = PTBXLDataset(test_df, records_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # Model with dropout
    print(f"\n🏗️  Building model with dropout={args.dropout}...")
    model = resnet18_1d(num_classes=len(SUPERCLASSES), include_patient_context=False,
                         dropout_rate=args.dropout)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")

    # Weighted loss for class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001, mode='max')

    # MLflow
    print("\n📊 Starting MLflow experiment...")
    mlflow.set_experiment("ecg-v2")

    with mlflow.start_run(run_name="resnet18-v2-improved"):
        mlflow.log_params({
            "model": "resnet18_1d_v2",
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "class_weighted": True,
            "device": str(device),
        })

        best_val_auc = 0.0
        best_model_path = output_dir / "best_model_v2.pth"

        print("\n🚀 Training...\n")
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            print("-" * 60)

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"Val AUC:    {val_auc:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
            }, step=epoch)

            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                print(f"✓ Best model saved (AUC: {best_val_auc:.4f})")

            # Early stopping check
            early_stopping(val_auc)
            if early_stopping.early_stop:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"   No improvement for {args.patience} epochs")
                break

            print()

        # Final evaluation
        print("=" * 60)
        print("FINAL TEST EVALUATION")
        print("=" * 60)

        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_auc, test_preds, test_labels = evaluate(model, test_loader,
                                                                 criterion, device)

        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test AUC:  {test_auc:.4f}")

        print("\nPer-class AUC:")
        per_class_aucs = {}
        for i, sc in enumerate(SUPERCLASSES):
            if len(np.unique(test_labels[:, i])) > 1:
                auc = roc_auc_score(test_labels[:, i], test_preds[:, i])
                per_class_aucs[sc] = auc
                print(f"  {sc}: {auc:.4f}")

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_auc": test_auc,
            "best_val_auc": best_val_auc,
            **{f"test_auc_{k}": v for k, v in per_class_aucs.items()}
        })

        # Log model
        mlflow.pytorch.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id

        print("\n✅ Training complete!")
        print(f"   Model saved to: {best_model_path}")
        print(f"   MLflow run ID: {run_id}")

        # Auto-register if good enough
        if args.auto_register and test_auc > 0.90:
            print(f"\n📦 Auto-registering model (AUC {test_auc:.4f} > 0.90)...")
            try:
                model_version = mlflow.register_model(
                    f"runs:/{run_id}/model",
                    "ecg-clinical-recommender"
                )
                print(f"✅ Registered as version {model_version.version}")

                # Promote to Staging (not Production - needs manual approval)
                client = mlflow.tracking.MlflowClient()
                client.set_registered_model_alias(
                    "ecg-clinical-recommender",
                    "candidate",
                    model_version.version
                )
                print("✅ Tagged as 'candidate' - ready for validation")

            except Exception as e:
                print(f"⚠️  Auto-registration failed: {e}")


if __name__ == "__main__":
    main()
