# ============================================================
# FILE: scripts/test_explainability.py
# ============================================================
"""
Test explainability features.

Usage:
    # Test Grad-CAM only (fast)
    python scripts/test_explainability.py --mode gradcam

    # Test with LLM explanations (slow, better quality)
    python scripts/test_explainability.py --mode llm

    # Test both
    python scripts/test_explainability.py --mode both

    # Test via API
    python scripts/test_explainability.py --mode api --use-llm
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))



def test_local_explainability(use_llm: bool = False):
    """Test explainability locally with predictor."""
    from src.ml.inference.predictor import get_predictor

    print("🔬 Testing Explainability Locally")
    print(f"   Mode: {'LLM-Enhanced' if use_llm else 'Grad-CAM Only'}")
    print("=" * 70)

    predictor = get_predictor()

    # Load a real ECG with MI
    data_dir = PROJECT_ROOT.parent / "data" / "ptb-xl" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    df = pd.read_csv(data_dir / "ptbxl_database.csv")

    # Find an MI case
    mi_df = df[df.scp_codes.str.contains('IMI|AMI')]
    if mi_df.empty:
        print("⚠️  No MI cases found, using random ECG")
        mi_case = df.sample(1).iloc[0]
    else:
        mi_case = mi_df.sample(1).iloc[0]

    record_path = data_dir / mi_case.filename_hr
    record = wfdb.rdsamp(str(record_path.with_suffix('')))
    signal = record[0].T

    print(f"\nPatient: {mi_case.ecg_id}")
    print(f"Age: {mi_case.age}, Sex: {'M' if mi_case.sex == 1 else 'F'}")
    print(f"True labels: {mi_case.scp_codes}")

    # Predict with explanation
    print(f"\n{'Generating LLM explanation (may take 5-10s)...' if use_llm else 'Generating Grad-CAM explanation...'}")

    result = predictor.predict(
        signal,
        explain=True,
        use_llm=use_llm,
        patient_age=int(mi_case.age) if pd.notna(mi_case.age) else None,
        patient_sex='M' if mi_case.sex == 1 else 'F',
    )

    print(f"\n📊 Predicted Diagnosis: {result['diagnoses'][0]['diagnosis']}")
    print(f"   Confidence: {result['diagnoses'][0]['confidence']:.2%}")

    if 'explanation' in result:
        exp = result['explanation']
        print("\n🔍 Explanation:")
        print(f"   Important leads: {', '.join(exp['important_leads'])}")
        print(f"   Time range: {exp['important_time_range']}")

        print("\n   Lead importance scores:")
        for lead, score in list(exp['lead_importance_scores'].items())[:6]:
            bars = '█' * int(score * 20)
            print(f"      {lead:3s}: {bars:<20s} {score:.2f}")

        print("\n   Clinical description:")
        print(f"      {exp['description']}")

        if 'llm_description' in exp:
            print("\n   📝 LLM-Enhanced Explanation:")
            print(f"      {exp['llm_description']}")
    else:
        print("\n⚠️  No explanation generated")


def test_api_explainability(use_llm: bool = False, api_url: str = "http://localhost:8000"):
    """Test explainability via API."""
    print("\n" + "=" * 70)
    print("🌐 Testing Explainability via API")
    print(f"   URL: {api_url}")
    print(f"   Mode: {'LLM-Enhanced' if use_llm else 'Grad-CAM Only'}")
    print("=" * 70)

    # Load sample ECG
    data_dir = PROJECT_ROOT.parent / "data" / "ptb-xl" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    mi_case = df[df.scp_codes.str.contains('IMI|AMI')].sample(1).iloc[0]

    record_path = data_dir / mi_case.filename_hr
    record = wfdb.rdsamp(str(record_path.with_suffix('')))
    signal = record[0].T

    print("\nSending request...")

    # Make API request
    response = requests.post(
        f'{api_url}/predict',
        json={
            'ecg_signal': signal.tolist(),
            'patient_age': int(mi_case.age) if pd.notna(mi_case.age) else None,
            'patient_sex': 'M' if mi_case.sex == 1 else 'F',
            'explain': True,
            'use_llm': use_llm,
        },
        timeout=60  # Longer timeout for LLM
    )

    if response.status_code == 200:
        result = response.json()

        print("\n✅ API Response received")
        print(f"   Processing time: {result['processing_time_ms']:.0f}ms")
        print(f"\n📊 Diagnosis: {result['diagnoses'][0]['diagnosis']}")
        print(f"   Confidence: {result['diagnoses'][0]['confidence']:.2%}")

        if result.get('explanation'):
            exp = result['explanation']
            print("\n🔍 Explanation:")
            print(f"   Important leads: {', '.join(exp['important_leads'])}")
            print(f"   Time range: {exp['important_time_range']}")
            print(f"   Description: {exp['description']}")
        else:
            print("\n⚠️  No explanation in response")
    else:
        print(f"\n❌ API request failed: {response.status_code}")
        print(response.text)


def main():
    parser = argparse.ArgumentParser(description="Test ECG explainability features")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['gradcam', 'llm', 'both', 'api'],
        default='gradcam',
        help='Test mode: gradcam (fast), llm (slow), both, or api'
    )
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use LLM for explanations (only applies to api mode)'
    )
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='API URL for testing'
    )

    args = parser.parse_args()

    print("🧪 ECG Explainability Test Suite")
    print()

    if args.mode == 'gradcam':
        test_local_explainability(use_llm=False)

    elif args.mode == 'llm':
        print("⚠️  First run will download BioMistral-7B-GGUF (~4GB)")
        print("   This may take 1-2 minutes...")
        input("Press Enter to continue...")
        test_local_explainability(use_llm=True)

    elif args.mode == 'both':
        print("Testing Grad-CAM first...")
        test_local_explainability(use_llm=False)

        print("\n\nNow testing with LLM...")
        print("⚠️  First run will download BioMistral-7B-GGUF (~4GB)")
        input("Press Enter to continue...")
        test_local_explainability(use_llm=True)

    elif args.mode == 'api':
        print("Make sure API is running:")
        print(f"  uvicorn src.api.main:app --reload --port {args.api_url.split(':')[-1]}")
        input("Press Enter when ready...")
        test_api_explainability(use_llm=args.use_llm, api_url=args.api_url)

    print("\n" + "=" * 70)
    print("✅ Explainability tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
