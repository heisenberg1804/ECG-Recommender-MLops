# ============================================================
# FILE: src/monitoring/drift_detector.py
# ============================================================
"""
Drift detection for ECG prediction system.

Monitors:
1. Input drift - ECG signal distribution changes
2. Output drift - Prediction distribution changes
3. Concept drift - Model performance degradation
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
)
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report


class ECGDriftDetector:
    """Detects drift in ECG predictions."""

    def __init__(
        self,
        db_url: str,
        reference_window_days: int = 7,
        current_window_hours: int = 24,
        drift_threshold: float = 0.1,
    ):
        """
        Initialize drift detector.

        Args:
            db_url: PostgreSQL connection string
            reference_window_days: Days of historical data as reference
            current_window_hours: Hours of recent data to compare
            drift_threshold: Threshold for drift detection (0-1)
        """
        self.db_url = db_url
        self.reference_window_days = reference_window_days
        self.current_window_hours = current_window_hours
        self.drift_threshold = drift_threshold

    async def get_predictions_df(
        self, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """Fetch predictions from database as DataFrame."""
        conn = await asyncpg.connect(self.db_url)

        rows = await conn.fetch(
            """
            SELECT
                id,
                ecg_id,
                created_at,
                patient_age,
                patient_sex,
                model_version,
                processing_time_ms,
                diagnoses,
                recommendations
            FROM predictions
            WHERE created_at >= $1 AND created_at < $2
            ORDER BY created_at
            """,
            start_time,
            end_time
        )

        await conn.close()

        if not rows:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in rows])

        # Parse JSONB fields
        df['diagnoses'] = df['diagnoses'].apply(json.loads)
        df['recommendations'] = df['recommendations'].apply(json.loads)

        # Extract features for drift detection
        df['num_diagnoses'] = df['diagnoses'].apply(len)
        df['num_recommendations'] = df['recommendations'].apply(len)
        df['max_confidence'] = df['diagnoses'].apply(
            lambda d: max([x['confidence'] for x in d]) if d else 0.0
        )
        df['avg_confidence'] = df['diagnoses'].apply(
            lambda d: np.mean([x['confidence'] for x in d]) if d else 0.0
        )

        # Extract primary diagnosis
        df['primary_diagnosis'] = df['diagnoses'].apply(
            lambda d: d[0]['diagnosis'] if d else 'NONE'
        )

        # Has urgent/immediate action?
        df['has_urgent_action'] = df['recommendations'].apply(
            lambda r: any(x['urgency'] in ['urgent', 'immediate'] for x in r)
        )

        return df

    async def detect_input_drift(self) -> dict[str, Any]:
        """
        Detect input drift in patient demographics and processing times.

        Returns drift report comparing reference vs current window.
        """
        now = datetime.utcnow()

        # Reference: 7 days ago
        ref_start = now - timedelta(days=self.reference_window_days + 1)
        ref_end = now - timedelta(days=1)

        # Current: Last 24 hours
        curr_start = now - timedelta(hours=self.current_window_hours)
        curr_end = now

        print("📊 Input Drift Detection")
        print(f"   Reference: {ref_start} to {ref_end}")
        print(f"   Current:   {curr_start} to {curr_end}")

        reference_df = await self.get_predictions_df(ref_start, ref_end)
        current_df = await self.get_predictions_df(curr_start, curr_end)

        if reference_df.empty or current_df.empty:
            return {
                'drift_detected': False,
                'message': 'Insufficient data for drift detection',
                'reference_count': len(reference_df),
                'current_count': len(current_df),
            }

        print(f"   Reference samples: {len(reference_df)}")
        print(f"   Current samples:   {len(current_df)}")

        # Columns to monitor
        numerical_features = ['patient_age', 'processing_time_ms']
        categorical_features = ['patient_sex', 'model_version']

        # Create Evidently report
        column_mapping = ColumnMapping(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        report = Report(metrics=[
            DataDriftPreset(drift_share=self.drift_threshold),
            DatasetDriftMetric(),
            ColumnDriftMetric(column_name='patient_age'),
            ColumnDriftMetric(column_name='processing_time_ms'),
        ])

        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping
        )

        # Extract results
        results = report.as_dict()

        drift_summary = {
            'drift_detected': results['metrics'][1]['result']['dataset_drift'],
            'drift_share': results['metrics'][1]['result']['drift_share'],
            'drifted_features': [],
            'reference_count': len(reference_df),
            'current_count': len(current_df),
            'timestamp': now.isoformat(),
        }

        # Check which features drifted
        for metric in results['metrics']:
            if metric['metric'] == 'ColumnDriftMetric':
                col_name = metric['result']['column_name']
                is_drifted = metric['result']['drift_detected']
                drift_score = metric['result'].get('drift_score', 0)

                if is_drifted:
                    drift_summary['drifted_features'].append({
                        'feature': col_name,
                        'drift_score': drift_score,
                    })

        return drift_summary

    async def detect_prediction_drift(self) -> dict[str, Any]:
        """
        Detect prediction drift - changes in model outputs.

        Monitors:
        - Diagnosis distribution changes
        - Confidence score distribution
        - Urgency level distribution
        """
        now = datetime.utcnow()

        ref_start = now - timedelta(days=self.reference_window_days + 1)
        ref_end = now - timedelta(days=1)
        curr_start = now - timedelta(hours=self.current_window_hours)
        curr_end = now

        print("📊 Prediction Drift Detection")

        reference_df = await self.get_predictions_df(ref_start, ref_end)
        current_df = await self.get_predictions_df(curr_start, curr_end)

        if reference_df.empty or current_df.empty:
            return {
                'drift_detected': False,
                'message': 'Insufficient data',
            }

        # Target features
        target_features = [
            'num_diagnoses',
            'num_recommendations',
            'max_confidence',
            'avg_confidence',
            'primary_diagnosis',
        ]

        column_mapping = ColumnMapping(
            target='primary_diagnosis',
            numerical_features=['num_diagnoses', 'num_recommendations',
                                 'max_confidence', 'avg_confidence'],
            categorical_features=['primary_diagnosis'],
        )

        report = Report(metrics=[
            TargetDriftPreset(),
            ColumnDriftMetric(column_name='max_confidence'),
            ColumnDriftMetric(column_name='primary_diagnosis'),
        ])

        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping
        )

        results = report.as_dict()

        return {
            'drift_detected': any(
                m['result'].get('drift_detected', False)
                for m in results['metrics']
                if 'drift_detected' in m['result']
            ),
            'metrics': results,
            'timestamp': now.isoformat(),
        }

    async def generate_drift_report(self, output_path: Path) -> None:
        """Generate HTML drift report."""
        now = datetime.utcnow()

        ref_start = now - timedelta(days=self.reference_window_days + 1)
        ref_end = now - timedelta(days=1)
        curr_start = now - timedelta(hours=self.current_window_hours)
        curr_end = now

        reference_df = await self.get_predictions_df(ref_start, ref_end)
        current_df = await self.get_predictions_df(curr_start, curr_end)

        if reference_df.empty or current_df.empty:
            print("⚠️  Insufficient data for report")
            return

        # Drop list columns that Evidently can't handle
        columns_to_drop = ['diagnoses', 'recommendations', 'ecg_id', 'id']
        reference_clean = reference_df.drop(columns=columns_to_drop, errors='ignore')
        current_clean = current_df.drop(columns=columns_to_drop, errors='ignore')

        # Define column mapping
        column_mapping = ColumnMapping(
            numerical_features=['patient_age', 'processing_time_ms', 'num_diagnoses',
                               'num_recommendations', 'max_confidence', 'avg_confidence'],
            categorical_features=['patient_sex', 'model_version', 'primary_diagnosis'],
            datetime_features=['created_at'],
        )

        # Comprehensive drift report
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
        ])

        report.run(
            reference_data=reference_clean,
            current_data=current_clean,
            column_mapping=column_mapping
        )

        # Save HTML report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(output_path))

        print(f"✅ Drift report saved to: {output_path}")


async def check_drift_and_alert(db_url: str, threshold: float = 0.1) -> dict[str, Any]:
    """
    Check for drift and return summary for alerting.

    Use this in a scheduled job (cron, Airflow, etc.)
    """
    detector = ECGDriftDetector(db_url=db_url, drift_threshold=threshold)

    input_drift = await detector.detect_input_drift()
    prediction_drift = await detector.detect_prediction_drift()

    summary = {
        'timestamp': datetime.utcnow().isoformat(),
        'input_drift': input_drift,
        'prediction_drift': prediction_drift,
        'alert': (
            input_drift.get('drift_detected', False) or
            prediction_drift.get('drift_detected', False)
        ),
    }

    if summary['alert']:
        print("🚨 DRIFT DETECTED!")
        print(f"   Input drift: {input_drift.get('drift_detected', False)}")
        print(f"   Prediction drift: {prediction_drift.get('drift_detected', False)}")
    else:
        print("✅ No drift detected")

    return summary
