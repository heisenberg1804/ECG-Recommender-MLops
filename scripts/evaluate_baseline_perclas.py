#!/usr/bin/env python3
"""
Evaluate baseline model and log per-class metrics to MLflow.
No retraining needed - just evaluation!
"""
import ast
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.train import PTBXLDataset  # Reuse existing
from src.ml.models.resnet1d import resnet18_1d

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))



# Load test data
data_dir = Path('./data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
df = pd.read_csv(data_dir / 'ptbxl_database.csv')
df['scp_codes_dict'] = df.scp_codes.apply(lambda x: ast.literal_eval(x))

# Same preprocessing as training
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

df['superclass_labels'] = df.scp_codes_dict.apply(lambda x: get_superclass_labels(x, SUPERCLASS_MAP))
df = df[df.superclass_labels.apply(len) > 0].reset_index(drop=True)
df['labels'] = df.superclass_labels.apply(encode_labels)

# Get test split
_, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Load baseline model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = resnet18_1d(num_classes=5)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Evaluate
test_dataset = PTBXLDataset(test_df, data_dir / 'records500')
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating baseline"):
        signals = batch['signal'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(signals)
        probs = torch.sigmoid(outputs)

        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# Calculate per-class AUC
print("\nPer-class AUC (Baseline Model):")
for i, sc in enumerate(SUPERCLASSES):
    if len(np.unique(all_labels[:, i])) > 1:
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        print(f"  {sc}: {auc:.4f}")

        # Log to MLflow (update existing run)
        # Find the baseline run
        exp = mlflow.get_experiment_by_name('ecg-baseline')
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

        if not runs.empty:
            run_id = runs.iloc[0]['run_id']

            # Log per-class metric
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric(f"test_auc_{sc}", auc)

print("\n✅ Per-class metrics added to baseline run!")
print("Re-run comparison script to see updated charts")
