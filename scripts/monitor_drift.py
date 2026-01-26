#!/usr/bin/env python3
# ============================================================
# FILE: scripts/monitor_drift.py
# ============================================================
"""
Monitor drift in ECG predictions.

Usage:
    # Generate drift report
    python scripts/monitor_drift.py --report

    # Check drift and print summary
    python scripts/monitor_drift.py --check

    # Run as continuous monitor (every hour)
    python scripts/monitor_drift.py --monitor --interval 3600
"""
import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from src.monitoring.drift_detector import ECGDriftDetector, check_drift_and_alert

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def generate_report(db_url: str, output_dir: Path):
    """Generate HTML drift report."""
    detector = ECGDriftDetector(db_url=db_url)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"drift_report_{timestamp}.html"

    print("📊 Generating drift report...")
    await detector.generate_drift_report(output_path)
    print(f"\n✅ Report available at: {output_path}")
    print("   Open in browser to view detailed drift analysis")


async def check_drift(db_url: str, threshold: float):
    """Check for drift and print summary."""
    print("🔍 Checking for drift...")
    summary = await check_drift_and_alert(db_url, threshold)

    print(f"\n{'='*60}")
    print("DRIFT SUMMARY")
    print(f"{'='*60}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Alert: {'🚨 YES' if summary['alert'] else '✅ NO'}")

    if summary['input_drift'].get('drift_detected'):
        print("\n📥 Input Drift Detected:")
        print(f"   Drift share: {summary['input_drift']['drift_share']:.2%}")
        for feat in summary['input_drift'].get('drifted_features', []):
            print(f"   - {feat['feature']}: {feat['drift_score']:.3f}")

    if summary['prediction_drift'].get('drift_detected'):
        print("\n📤 Prediction Drift Detected")

    return summary


async def monitor_continuously(db_url: str, threshold: float, interval: int):
    """Continuously monitor for drift."""
    print(f"🔄 Starting continuous drift monitoring (every {interval}s)...")
    print("   Press Ctrl+C to stop")

    while True:
        try:
            await check_drift(db_url, threshold)
            print(f"\n⏰ Next check in {interval} seconds...")
            await asyncio.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
            break
        except Exception as e:
            print(f"\n⚠️  Error during drift check: {e}")
            await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Monitor drift in ECG predictions")
    parser.add_argument(
        '--db-url',
        type=str,
        default='postgresql://ecg_user:ecg_password_dev@localhost:5432/ecg_predictions',
        help='PostgreSQL connection URL'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='Drift detection threshold (0-1)'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate HTML drift report'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check for drift once'
    )
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Monitor continuously'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=3600,
        help='Monitoring interval in seconds (default: 3600 = 1 hour)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports/drift'),
        help='Output directory for reports'
    )

    args = parser.parse_args()

    if args.report:
        asyncio.run(generate_report(args.db_url, args.output_dir))
    elif args.check:
        asyncio.run(check_drift(args.db_url, args.threshold))
    elif args.monitor:
        asyncio.run(monitor_continuously(args.db_url, args.threshold, args.interval))
    else:
        print("❌ Please specify --report, --check, or --monitor")
        parser.print_help()


if __name__ == "__main__":
    main()
