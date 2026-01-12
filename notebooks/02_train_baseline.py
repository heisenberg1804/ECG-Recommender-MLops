# ============================================================
# FILE: notebooks/02_train_baseline.ipynb
# ============================================================
# This is a Python script version - convert to notebook with:
# jupytext --to notebook 02_train_baseline.py

"""
Train baseline ResNet-18 on PTB-XL dataset.

Goal: Get a working model quickly that can:
1. Classify ECG signals into diagnostic categories
2. Map diagnoses to clinical actions
3. Be deployed in the FastAPI service
"""

# %% [markdown]
# # Baseline ECG Clinical Action Recommender
#
# Train ResNet-18 on PTB-XL for multi-label classification of diagnostic superclasses.
# Then map diagnoses to clinical actions.

# %% Imports
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

# Add src to path
sys.path.append(str(Path.cwd().parent))

# Import our model
from src.ml.models.resnet1d import resnet18_1d

# %% [markdown]
# ## 1. Data Loading

# %% Configuration
DATA_DIR = Path("../data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
RECORDS_DIR = DATA_DIR / "records500"  # Use 500 Hz
DATABASE_FILE = DATA_DIR / "ptbxl_database.csv"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_WORKERS = 4

print(f"Using device: {DEVICE}")

# %% Load metadata
df = pd.read_csv(DATABASE_FILE)
print(f"Total records: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())

# %% Parse SCP codes
# PTB-XL stores diagnostic codes as string dictionaries
df['scp_codes_dict'] = df.scp_codes.apply(lambda x: ast.literal_eval(x))

# Extract diagnostic superclass (main diagnostic category)
def get_superclass_labels(scp_dict, superclass_mapping):
    """Extract superclass labels from SCP codes."""
    labels = set()
    for code in scp_dict.keys():
        if code in superclass_mapping:
            labels.add(superclass_mapping[code])
    return labels

# Load superclass mapping (from PTB-XL paper)
# NORM: Normal ECG
# MI: Myocardial Infarction
# STTC: ST/T Change
# CD: Conduction Disturbance
# HYP: Hypertrophy

# Simplified mapping of common codes to superclasses
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

df['superclass_labels'] = df.scp_codes_dict.apply(
    lambda x: get_superclass_labels(x, SUPERCLASS_MAP)
)

# Filter to records with at least one superclass label
df = df[df.superclass_labels.apply(len) > 0].reset_index(drop=True)
print(f"\nRecords with superclass labels: {len(df)}")

# Create multi-label encoding
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
def encode_labels(label_set):
    return [1 if sc in label_set else 0 for sc in SUPERCLASSES]

df['labels'] = df.superclass_labels.apply(encode_labels)

# Check label distribution
label_counts = df.labels.apply(lambda x: sum(x)).value_counts().sort_index()
print("\nLabel distribution (# of labels per ECG):")
print(label_counts)

print("\nClass frequencies:")
for i, sc in enumerate(SUPERCLASSES):
    count = sum(df.labels.apply(lambda x: x[i]))
    print(f"  {sc}: {count} ({count/len(df)*100:.1f}%)")

# %% Train/val/test split
# Use stratified split on the most common label
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.labels.apply(lambda x: x[0])
)
train_df, val_df = train_test_split(
    train_df, test_size=0.1, random_state=42, stratify=train_df.labels.apply(lambda x: x[0])
)

print("\nSplit sizes:")
print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")

# %% [markdown]
# ## 2. Dataset & DataLoader

# %% ECG Dataset
class PTBXLDataset(Dataset):
    """PTB-XL dataset for ECG signals."""

    def __init__(self, df, records_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.records_dir = Path(records_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load ECG signal
        record_path = self.records_dir / row.filename_hr  # Use high-res (500Hz)
        record = wfdb.rdsamp(str(record_path.with_suffix('')))
        signal = record[0]  # (5000, 12) numpy array

        # Transpose to (12, 5000) for Conv1D
        signal = signal.T.astype(np.float32)

        # Normalize (simple z-score per lead)
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (
            signal.std(axis=1, keepdims=True) + 1e-8
        )

        # Get labels
        labels = np.array(row.labels, dtype=np.float32)

        # Patient context (for future use)
        age = row.age / 100.0  # Normalize to [0, 1] roughly
        sex = 1.0 if row.sex == 1 else 0.0  # Male=1, Female=0
        patient_context = np.array([age, sex, 0.0], dtype=np.float32)  # 3rd feature placeholder

        if self.transform:
            signal = self.transform(signal)

        return {
            'signal': torch.from_numpy(signal),
            'labels': torch.from_numpy(labels),
            'patient_context': torch.from_numpy(patient_context),
        }

# Create datasets
train_dataset = PTBXLDataset(train_df, RECORDS_DIR)
val_dataset = PTBXLDataset(val_df, RECORDS_DIR)
test_dataset = PTBXLDataset(test_df, RECORDS_DIR)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

print("\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches:   {len(val_loader)}")
print(f"  Test batches:  {len(test_loader)}")

# Test loading one batch
sample_batch = next(iter(train_loader))
print("\nSample batch shapes:")
print(f"  Signal: {sample_batch['signal'].shape}")
print(f"  Labels: {sample_batch['labels'].shape}")
print(f"  Context: {sample_batch['patient_context'].shape}")

# %% [markdown]
# ## 3. Model Training

# %% Initialize model
model = resnet18_1d(num_classes=len(SUPERCLASSES), include_patient_context=False)
model = model.to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nModel: ResNet-18 1D")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# %% Loss function and optimizer
# Use BCEWithLogitsLoss for multi-label classification
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

# %% Training loop
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        signals = batch['signal'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
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

            # Get probabilities
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Calculate metrics
    avg_loss = total_loss / len(loader)

    # AUC-ROC per class
    aucs = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:  # Only if both classes present
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)

    macro_auc = np.mean(aucs) if aucs else 0.0

    return avg_loss, macro_auc, all_preds, all_labels

# %% Start MLflow run
mlflow.set_experiment("ecg-baseline")

with mlflow.start_run(run_name="resnet18-baseline"):
    # Log parameters
    mlflow.log_params({
        "model": "resnet18_1d",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "device": str(DEVICE),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    })

    best_val_auc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val AUC:    {val_auc:.4f}")

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
        }, step=epoch)

        # Learning rate scheduling
        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "../models/best_model.pth")
            mlflow.log_metric("best_val_auc", best_val_auc)
            print(f"  ✓ Best model saved (AUC: {best_val_auc:.4f})")

    # %% Final evaluation on test set
    print("\n" + "="*50)
    print("FINAL TEST EVALUATION")
    print("="*50)

    # Load best model
    model.load_state_dict(torch.load("../models/best_model.pth"))

    test_loss, test_auc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test AUC:  {test_auc:.4f}")

    # Per-class AUC
    print("\nPer-class AUC:")
    for i, sc in enumerate(SUPERCLASSES):
        if len(np.unique(test_labels[:, i])) > 1:
            auc = roc_auc_score(test_labels[:, i], test_preds[:, i])
            print(f"  {sc}: {auc:.4f}")

    # Log test metrics
    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_auc": test_auc,
    })

    # Log model
    mlflow.pytorch.log_model(model, "model")

    print("\n✓ Model logged to MLflow")
    print(f"✓ Run ID: {mlflow.active_run().info.run_id}")

# %% [markdown]
# ## 4. Diagnostic → Action Mapping
#
# Define mapping from diagnostic superclasses to clinical actions

# %% Action mapping
DIAGNOSTIC_TO_ACTIONS = {
    'MI': [
        {
            'action': 'Activate cath lab',
            'urgency': 'immediate',
            'reasoning': 'Myocardial infarction detected - requires immediate intervention'
        },
        {
            'action': 'Administer aspirin 325mg',
            'urgency': 'immediate',
            'reasoning': 'Antiplatelet therapy for acute MI'
        },
        {
            'action': 'Order troponin stat',
            'urgency': 'immediate',
            'reasoning': 'Confirm cardiac injury with biomarkers'
        },
    ],
    'STTC': [
        {
            'action': 'Order troponin levels',
            'urgency': 'urgent',
            'reasoning': 'ST/T changes may indicate ischemia'
        },
        {
            'action': '12-lead ECG in 6 hours',
            'urgency': 'routine',
            'reasoning': 'Monitor for dynamic changes'
        },
        {
            'action': 'Cardiology consult',
            'urgency': 'urgent',
            'reasoning': 'Specialist review of ST/T abnormalities'
        },
    ],
    'CD': [
        {
            'action': 'Review medications',
            'urgency': 'routine',
            'reasoning': 'Conduction disturbance - check for QT-prolonging drugs'
        },
        {
            'action': 'Consider 24-hour Holter monitor',
            'urgency': 'routine',
            'reasoning': 'Assess for intermittent conduction blocks'
        },
    ],
    'HYP': [
        {
            'action': 'Order echocardiogram',
            'urgency': 'routine',
            'reasoning': 'Hypertrophy detected - assess cardiac function'
        },
        {
            'action': 'Blood pressure monitoring',
            'urgency': 'routine',
            'reasoning': 'Evaluate for hypertension as underlying cause'
        },
    ],
    'NORM': [
        {
            'action': 'Normal ECG - discharge with instructions',
            'urgency': 'routine',
            'reasoning': 'No acute abnormalities detected'
        },
    ],
}

print("\nDiagnostic → Action Mapping:")
for dx, actions in DIAGNOSTIC_TO_ACTIONS.items():
    print(f"\n{dx}:")
    for action in actions:
        print(f"  • {action['action']} ({action['urgency']})")

# %% Save mapping as JSON for API

mapping_file = Path("../src/ml/inference/action_mapping.json")
mapping_file.parent.mkdir(parents=True, exist_ok=True)

with open(mapping_file, 'w') as f:
    json.dump(DIAGNOSTIC_TO_ACTIONS, f, indent=2)

print(f"\n✓ Action mapping saved to {mapping_file}")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print("\nNext steps:")
print("1. Check MLflow UI (http://localhost:5000) for experiment tracking")
print("2. Model saved to: ../models/best_model.pth")
print("3. Ready to integrate into FastAPI service")
