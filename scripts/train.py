#!/usr/bin/env python3
# ============================================================
# FILE: scripts/train.py
# ============================================================
"""
Train baseline ResNet-18 on PTB-XL dataset.

Usage:
    python scripts/train.py
    # Or with custom config:
    python scripts/train.py --epochs 30 --batch-size 64
"""
import argparse
import ast
import json
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Now imports will work
from src.ml.models.resnet1d import resnet18_1d

# Add project root to path BEFORE any local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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
        signal = record[0]  # (5000, 12) numpy array

        # Transpose to (12, 5000) for Conv1D
        signal = signal.T.astype(np.float32)

        # Normalize (z-score per lead)
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (
            signal.std(axis=1, keepdims=True) + 1e-8
        )

        # Get labels
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
# Main Training Script
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, 
                       default='../data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3',
                       help='Path to PTB-XL dataset')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Where to save trained model')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None, 
                       help='Device (cpu/mps/cuda). Auto-detect if None.')
    args = parser.parse_args()
    # Setup paths
    data_dir = Path(args.data_dir)
    records_dir = data_dir / "records500"
    database_file = data_dir / "ptbxl_database.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check files exist
    if not database_file.exists():
        print(f"❌ Database file not found: {database_file}")
        print("   Please check your --data-dir path")
        return

    # Auto-detect device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"🖥️  Using device: {device}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📈 Learning rate: {args.lr}")
    print()

    # Load data
    print("📁 Loading PTB-XL database...")
    df = pd.read_csv(database_file)
    print(f"   Total records: {len(df)}")

    # Parse labels
    df['scp_codes_dict'] = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Superclass mapping
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

    print(f"   Records with labels: {len(df)}")
    print("\n   Class distribution:")
    for i, sc in enumerate(SUPERCLASSES):
        count = sum(df.labels.apply(lambda x: x[i]))
        print(f"      {sc}: {count} ({count/len(df)*100:.1f}%)")

    # Train/val/test split
    print("\n🔀 Splitting data...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=df.labels.apply(lambda x: x[0])
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=42,
        stratify=train_df.labels.apply(lambda x: x[0])
    )

    print(f"   Train: {len(train_df)}")
    print(f"   Val:   {len(val_df)}")
    print(f"   Test:  {len(test_df)}")

    # Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = PTBXLDataset(train_df, records_dir)
    val_dataset = PTBXLDataset(val_df, records_dir)
    test_dataset = PTBXLDataset(test_df, records_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    # Create model
    print("\n🏗️  Building model...")
    model = resnet18_1d(num_classes=len(SUPERCLASSES), include_patient_context=False)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # MLflow tracking
    print("\n📊 Starting MLflow experiment...")
    mlflow.set_experiment("ecg-baseline")

    with mlflow.start_run(run_name="resnet18-baseline"):
        # Log parameters
        mlflow.log_params({
            "model": "resnet18_1d",
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "num_epochs": args.epochs,
            "device": str(device),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
        })

        best_val_auc = 0.0
        best_model_path = output_dir / "best_model.pth"

        # Training loop
        print("\n🚀 Starting training...\n")
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            print("-" * 50)

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

            print()

        # Final test evaluation
        print("=" * 50)
        print("FINAL TEST EVALUATION")
        print("=" * 50)

        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_auc, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )

        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test AUC:  {test_auc:.4f}")

        print("\nPer-class AUC:")
        for i, sc in enumerate(SUPERCLASSES):
            if len(np.unique(test_labels[:, i])) > 1:
                auc = roc_auc_score(test_labels[:, i], test_preds[:, i])
                print(f"  {sc}: {auc:.4f}")

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_auc": test_auc,
            "best_val_auc": best_val_auc,
        })

        # Log model
        mlflow.pytorch.log_model(model, "model")

        print("\n✅ Training complete!")
        print(f"   Model saved to: {best_model_path}")
        print(f"   MLflow run ID: {mlflow.active_run().info.run_id}")

    # Save action mapping
    print("\n💾 Saving action mapping...")
    DIAGNOSTIC_TO_ACTIONS = {
        'MI': [
            {'action': 'Activate cath lab', 'urgency': 'immediate',
             'reasoning': 'Myocardial infarction detected - requires immediate intervention'},
            {'action': 'Administer aspirin 325mg', 'urgency': 'immediate',
             'reasoning': 'Antiplatelet therapy for acute MI'},
            {'action': 'Order troponin stat', 'urgency': 'immediate',
             'reasoning': 'Confirm cardiac injury with biomarkers'},
        ],
        'STTC': [
            {'action': 'Order troponin levels', 'urgency': 'urgent',
             'reasoning': 'ST/T changes may indicate ischemia'},
            {'action': '12-lead ECG in 6 hours', 'urgency': 'routine',
             'reasoning': 'Monitor for dynamic changes'},
            {'action': 'Cardiology consult', 'urgency': 'urgent',
             'reasoning': 'Specialist review of ST/T abnormalities'},
        ],
        'CD': [
            {'action': 'Review medications', 'urgency': 'routine',
             'reasoning': 'Conduction disturbance - check for QT-prolonging drugs'},
            {'action': 'Consider 24-hour Holter monitor', 'urgency': 'routine',
             'reasoning': 'Assess for intermittent conduction blocks'},
        ],
        'HYP': [
            {'action': 'Order echocardiogram', 'urgency': 'routine',
             'reasoning': 'Hypertrophy detected - assess cardiac function'},
            {'action': 'Blood pressure monitoring', 'urgency': 'routine',
             'reasoning': 'Evaluate for hypertension as underlying cause'},
        ],
        'NORM': [
            {'action': 'Normal ECG - discharge with instructions', 'urgency': 'routine',
             'reasoning': 'No acute abnormalities detected'},
        ],
    }

    mapping_path = PROJECT_ROOT / "src" / "ml" / "inference" / "action_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    with open(mapping_path, 'w') as f:
        json.dump(DIAGNOSTIC_TO_ACTIONS, f, indent=2)

    print(f"   Action mapping saved to: {mapping_path}")

    print("\n🎉 All done! Next steps:")
    print("   1. Check MLflow UI: http://localhost:5000")
    print("   2. Test inference with: python scripts/test_inference.py")
    print("   3. Update FastAPI endpoint and deploy")


if __name__ == "__main__":
    main()
