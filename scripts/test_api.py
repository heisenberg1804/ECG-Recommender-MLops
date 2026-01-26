#!/usr/bin/env python3
# ============================================================
# FILE: scripts/test_api.py
# ============================================================
"""
Test the FastAPI endpoint with real ECG data from PTB-XL.

Usage:
    # Terminal 1: Start API
    uvicorn src.api.main:app --reload
    # Terminal 2: Run test
    python scripts/test_api.py
"""
import ast
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_sample_ecgs(data_dir, num_samples=3):
    """Load sample ECGs from PTB-XL."""
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    df['scp_codes_dict'] = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    samples = []

    # Get one normal
    norm_df = df[df.scp_codes.str.contains('NORM')].sample(1)
    samples.append(('NORM - Normal ECG', norm_df.iloc[0]))

    # Get one MI
    mi_df = df[df.scp_codes.str.contains('IMI|AMI')].sample(1)
    samples.append(('MI - Myocardial Infarction', mi_df.iloc[0]))

    # Get one STTC
    sttc_df = df[df.scp_codes.str.contains('STTC|NST_')].sample(1)
    samples.append(('STTC - ST/T Changes', sttc_df.iloc[0]))

    return samples


def test_api_with_ecg(api_url, signal, true_label, patient_info):
    """Test API with a single ECG."""
    print(f"\n{'='*80}")
    print(f"Testing: {true_label}")
    print(f"Patient: ID={patient_info['ecg_id']}, Age={patient_info['age']}, Sex={patient_info['sex']}")
    print(f"{'='*80}")

    # Prepare request
    payload = {
        "ecg_signal": signal.tolist(),
        "patient_age": int(patient_info['age']) if pd.notna(patient_info['age']) else None,
        "patient_sex": patient_info['sex'] if patient_info['sex'] in ['M', 'F'] else None,
        "sampling_rate": 500
    }

    # Make request
    start_time = time.time()
    try:
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        request_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()

            print("\n✅ Request successful!")
            print(f"   Request time: {request_time:.0f}ms")
            print(f"   Processing time: {result['processing_time_ms']:.0f}ms")
            print(f"   ECG ID: {result['ecg_id']}")

            # Display diagnoses
            print("\n📊 Predicted Diagnoses:")
            if result['diagnoses']:
                for dx in result['diagnoses']:
                    confidence = dx['confidence'] * 100
                    print(f"   • {dx['diagnosis']}: {confidence:.1f}% confidence")
            else:
                print("   No significant abnormalities detected")

            # Display recommendations
            print("\n💊 Clinical Recommendations:")
            for i, rec in enumerate(result['recommendations'], 1):
                urgency_emoji = {
                    'immediate': '🚨',
                    'urgent': '⚠️',
                    'routine': 'ℹ️'
                }
                emoji = urgency_emoji.get(rec['urgency'], 'ℹ️')
                confidence = rec['confidence'] * 100

                print(f"\n   {i}. {emoji} {rec['action']}")
                print(f"      Confidence: {confidence:.1f}%")
                print(f"      Urgency: {rec['urgency'].upper()}")
                print(f"      Reasoning: {rec['reasoning']}")

            return True

        else:
            print("\n❌ Request failed!")
            print(f"   Status code: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("\n❌ Connection failed!")
        print("   Make sure API is running: uvicorn src.api.main:app --reload")
        return False
    except Exception as e:
        print("\n❌ Request failed with error:")
        print(f"   {type(e).__name__}: {e}")
        return False


def test_model_info(api_url):
    """Test the /model/info endpoint."""
    print(f"\n{'='*80}")
    print("Testing /model/info endpoint")
    print(f"{'='*80}")

    try:
        response = requests.get(f"{api_url}/model/info", timeout=10)

        if response.status_code == 200:
            info = response.json()
            print("\n✅ Model Info Retrieved:")
            print(f"   Model Type: {info['model_type']}")
            print(f"   Classes: {', '.join(info['classes'])}")
            print(f"   Device: {info['device']}")
            print(f"   Model Path: {info['model_path']}")
            return True
        else:
            print(f"\n❌ Request failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"\n❌ Request failed: {e}")
        return False


def main():
    print("🧪 Testing ECG Clinical Action Recommender API")
    print("="*80)

    # Configuration
    API_URL = "http://localhost:8000"
    DATA_DIR = PROJECT_ROOT.parent / "data" / "ptb-xl" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

    # Test health check
    print("\n🔍 Checking if API is running...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ API is running!")
        else:
            print(f"   ⚠️  API responded with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to API!")
        print("   Please start the API first:")
        print("      uvicorn src.api.main:app --reload")
        return

    # Test model info endpoint
    test_model_info(API_URL)

    # Load sample ECGs
    print("\n📁 Loading sample ECGs from PTB-XL...")
    samples = load_sample_ecgs(DATA_DIR)
    print(f"   Loaded {len(samples)} samples")

    # Test each sample
    success_count = 0
    for true_label, row in samples:
        # Load ECG signal
        record_path = DATA_DIR / row.filename_hr
        record = wfdb.rdsamp(str(record_path.with_suffix('')))
        signal = record[0].T  # (12, 5000)

        patient_info = {
            'ecg_id': row.ecg_id,
            'age': row.age,
            'sex': 'M' if row.sex == 1 else 'F'
        }

        # Test API
        success = test_api_with_ecg(API_URL, signal, true_label, patient_info)
        if success:
            success_count += 1

        # Brief pause between requests
        time.sleep(0.5)

    # Summary
    print(f"\n{'='*80}")
    print("📊 Test Summary")
    print(f"{'='*80}")
    print(f"   Total tests: {len(samples)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(samples) - success_count}")

    if success_count == len(samples):
        print("\n   ✅ All tests passed!")
    else:
        print("\n   ⚠️  Some tests failed")

    print(f"\n{'='*80}")
    print("Next steps:")
    print("  1. Check API logs for any errors")
    print("  2. View Swagger docs: http://localhost:8000/docs")
    print("  3. Deploy to Docker/Kubernetes")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
