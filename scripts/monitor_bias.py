#!/usr/bin/env python3
# ============================================================
# FILE: scripts/monitor_bias.py
# ============================================================
"""
Monitor fairness and bias in ECG predictions.

Usage:
    python scripts/monitor_bias.py --check
    python scripts/monitor_bias.py --report
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

from src.monitoring.bias_analyzer import BiasAnalyzer, check_bias_and_alert

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def check_bias(db_url: str):
    """Check for bias and print summary."""
    print("🔍 Checking for bias across demographics...")
    report = await check_bias_and_alert(db_url)

    print(f"\n{'='*70}")
    print("BIAS ANALYSIS REPORT")
    print(f"{'='*70}")
    print(f"{report['summary']}")

    if report['bias_detected']:
        print("\n⚠️  WARNINGS:")
        for warning in report['warnings']:
            print(f"   • {warning}")

    # Demographic parity details
    if 'sex_parity' in report['demographic_parity']:
        sex = report['demographic_parity']['sex_parity']
        print("\n📊 Sex-based Demographic Parity:")
        print(f"   Male urgent rate:   {sex.get('male_urgent_rate', 0):.2%} (n={sex.get('male_count', 0)})")
        print(f"   Female urgent rate: {sex.get('female_urgent_rate', 0):.2%} (n={sex.get('female_count', 0)})")
        print(f"   Parity ratio:       {sex.get('parity_ratio', 0):.3f} (threshold: 0.8)")
        print(f"   Status: {'✅ PASS' if sex.get('passes') else '❌ FAIL'}")

    if 'age_parity' in report['demographic_parity']:
        age = report['demographic_parity']['age_parity']
        print("\n📊 Age-based Demographic Parity:")
        for group, data in age.items():
            if group not in ['parity_ratio', 'passes']:
                print(f"   {group}: {data.get('urgent_rate', 0):.2%} (n={data.get('count', 0)})")
        if 'parity_ratio' in age:
            print(f"   Parity ratio: {age['parity_ratio']:.3f} (threshold: 0.8)")
            print(f"   Status: {'✅ PASS' if age.get('passes') else '❌ FAIL'}")

    # Prediction distributions
    if 'by_sex' in report['prediction_distribution']:
        print("\n📈 Diagnosis Distribution by Sex:")
        for sex, data in report['prediction_distribution']['by_sex'].items():
            print(f"\n   {sex} (n={data['count']}):")
            print(f"      Avg confidence: {data['avg_confidence']:.2%}")
            print("      Top diagnoses:")
            for dx, rate in sorted(
                data['diagnosis_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]:
                print(f"         {dx}: {rate:.2%}")


async def generate_report(db_url: str, output_path: Path):
    """Generate JSON bias report."""
    analyzer = BiasAnalyzer(db_url=db_url)
    report = await analyzer.generate_bias_report()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python native types for JSON
    def convert_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    report_clean = convert_types(report)

    with open(output_path, 'w') as f:
        json.dump(report_clean, f, indent=2)

    print(f"✅ Bias report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Monitor bias in ECG predictions")
    parser.add_argument(
        '--db-url',
        type=str,
        default='postgresql://ecg_user:ecg_password_dev@localhost:5432/ecg_predictions',
        help='PostgreSQL connection URL'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check for bias and print summary'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate JSON bias report'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('reports/bias/bias_report.json'),
        help='Output path for JSON report'
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=7,
        help='Days of data to analyze'
    )

    args = parser.parse_args()

    if args.check:
        asyncio.run(check_bias(args.db_url))
    elif args.report:
        asyncio.run(generate_report(args.db_url, args.output))
    else:
        print("❌ Please specify --check or --report")
        parser.print_help()


if __name__ == "__main__":
    main()
