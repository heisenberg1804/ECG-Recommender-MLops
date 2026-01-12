#!/usr/bin/env python3
# ============================================================
# FILE: scripts/test_inference.py
# ============================================================
"""
Test the trained model with real ECG data.

Usage:
    python scripts/test_inference.py
"""
import ast
import sys
from pathlib import Path

import pandas as pd
import wfdb
from src.ml.inference.predictor import get_predictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_sample_ecg(data_dir, num_samples=5):
    """Load sample ECGs from PTB-XL."""
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    df['scp_codes_dict'] = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Get diverse samples (one from each class if possible)
    samples = []

    # Get one normal
    norm_samples = df[df.scp_codes.str.contains('NORM')].sample(1)
    samples.append(('NORM', norm_samples.iloc[0]))

    # Get one MI
    mi_samples = df[df.scp_codes.str.contains('IMI|AMI')].sample(1)
    samples.append(('MI', mi_samples.iloc[0]))

    # Get one STTC
    sttc_samples = df[df.scp_codes.str.contains('STTC|NST_')].sample(1)
    samples.append(('STTC', sttc_samples.iloc[0]))

    # Get one CD
    cd_samples = df[df.scp_codes.str.contains('IRBBB|CLBBB')].sample(1)
    samples.append(('CD', cd_samples.iloc[0]))

    # Get one HYP
    hyp_samples = df[df.scp_codes.str.contains('LVH|RVH')].sample(1)
    samples.append(('HYP', hyp_samples.iloc[0]))

    return samples


def test_single_ecg(predictor, signal, true_label, patient_id):
    """Test inference on a single ECG."""
    print(f"\n{'='*70}")
    print(f"Patient ID: {patient_id}")
    print(f"True Label: {true_label}")
    print(f"{'='*70}")

    # Run inference
    result = predictor.predict(signal, threshold=0.3, top_k=5)

    # Display diagnoses
    print("\n📊 Predicted Diagnoses:")
    if result['diagnoses']:
        for dx in result['diagnoses']:
            print(f"  • {dx['diagnosis']}: {dx['confidence']:.2%} confidence")
    else:
        print("  No significant abnormalities detected")

    # Display recommendations
    print("\n💊 Clinical Recommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        urgency_emoji = {
            'immediate': '🚨',
            'urgent': '⚠️',
            'routine': 'ℹ️'
        }
        emoji = urgency_emoji.get(rec['urgency'], 'ℹ️')
        print(f"\n  {i}. {emoji} {rec['action']}")
        print(f"     Confidence: {rec['confidence']:.2%}")
        print(f"     Urgency: {rec['urgency'].upper()}")
        print(f"     Reasoning: {rec['reasoning']}")


def main():
    print("🔬 Testing ECG Clinical Action Predictor")
    print("=" * 70)

    # Setup paths
    model_path = PROJECT_ROOT / "models" / "best_model.pth"
    action_mapping_path = PROJECT_ROOT / "src" / "ml" / "inference" / "action_mapping.json"
    data_dir = PROJECT_ROOT.parent / "data" / "ptb-xl" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Run training first: python scripts/train.py")
        return

    print(f"✅ Loading model from: {model_path}")
    print(f"✅ Loading action mapping from: {action_mapping_path}")

    # Initialize predictor
    predictor = get_predictor(
        model_path=model_path,
        action_mapping_path=action_mapping_path
    )

    print(f"✅ Model loaded on device: {predictor.device}")
    print()

    # Load sample ECGs
    print("📁 Loading sample ECGs from PTB-XL...")
    samples = load_sample_ecg(data_dir)

    # Test each sample
    for true_label, row in samples:
        # Load ECG signal
        record_path = data_dir / row.filename_hr
        record = wfdb.rdsamp(str(record_path.with_suffix('')))
        signal = record[0].T  # (12, 5000)

        # Run inference
        test_single_ecg(
            predictor=predictor,
            signal=signal,
            true_label=true_label,
            patient_id=row.ecg_id
        )

    print("\n" + "="*70)
    print("✅ Inference testing complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Update FastAPI endpoint with real model")
    print("  2. Test API with: curl -X POST http://localhost:8000/predict")
    print("  3. Deploy to Kubernetes")


if __name__ == "__main__":
    main()
