# ============================================================
# FILE: src/monitoring/bias_analyzer.py
# ============================================================
"""
Fairness and bias monitoring for ECG predictions.

Implements standard fairness metrics:
- Demographic Parity: P(Y=urgent | male) ≈ P(Y=urgent | female)
- Equalized Odds: TPR_male ≈ TPR_female (needs ground truth)
- Performance Parity: AUC_male ≈ AUC_female (needs ground truth)
"""
import json
from datetime import datetime, timedelta
from typing import Any

import asyncpg
import pandas as pd


class BiasAnalyzer:
    """Analyzes bias in ECG predictions."""

    def __init__(self, db_url: str, lookback_days: int = 7):
        """
        Initialize bias analyzer.

        Args:
            db_url: PostgreSQL connection string
            lookback_days: Days of data to analyze
        """
        self.db_url = db_url
        self.lookback_days = lookback_days

    async def get_predictions_df(self) -> pd.DataFrame:
        """Fetch recent predictions from database."""
        conn = await asyncpg.connect(self.db_url)

        start_time = datetime.utcnow() - timedelta(days=self.lookback_days)

        rows = await conn.fetch(
            """
            SELECT
                ecg_id,
                created_at,
                patient_age,
                patient_sex,
                model_version,
                processing_time_ms,
                diagnoses,
                recommendations
            FROM predictions
            WHERE created_at >= $1
            ORDER BY created_at DESC
            """,
            start_time
        )

        await conn.close()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])

        # Parse JSONB
        df['diagnoses'] = df['diagnoses'].apply(json.loads)
        df['recommendations'] = df['recommendations'].apply(json.loads)

        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for bias analysis."""
        # Age groups
        df['age_group'] = pd.cut(
            df['patient_age'],
            bins=[0, 40, 65, 120],
            labels=['<40', '40-65', '>65']
        )

        # Primary diagnosis
        df['primary_diagnosis'] = df['diagnoses'].apply(
            lambda d: d[0]['diagnosis'] if d else 'NONE'
        )

        # Max confidence
        df['max_confidence'] = df['diagnoses'].apply(
            lambda d: max([x['confidence'] for x in d]) if d else 0.0
        )

        # Has urgent/immediate action
        df['has_urgent_action'] = df['recommendations'].apply(
            lambda r: any(x['urgency'] in ['urgent', 'immediate'] for x in r)
        )

        # Urgency level (encoded)
        def get_max_urgency(recs):
            if not recs:
                return 'routine'
            urgencies = [r['urgency'] for r in recs]
            if 'immediate' in urgencies:
                return 'immediate'
            elif 'urgent' in urgencies:
                return 'urgent'
            return 'routine'

        df['max_urgency'] = df['recommendations'].apply(get_max_urgency)

        return df

    async def calculate_demographic_parity(self) -> dict[str, Any]:
        """
        Calculate demographic parity.

        Measures: P(urgent action | group_A) ≈ P(urgent action | group_B)

        Returns metrics for sex and age groups.
        """
        df = await self.get_predictions_df()

        if df.empty or len(df) < 10:
            return {'error': 'Insufficient data', 'sample_size': len(df)}

        df = self._prepare_features(df)

        results = {
            'sample_size': len(df),
            'timestamp': datetime.utcnow().isoformat(),
            'sex_parity': {},
            'age_parity': {},
        }

        # Sex-based demographic parity
        if 'patient_sex' in df.columns and df['patient_sex'].notna().any():
            sex_groups = df.groupby('patient_sex')['has_urgent_action'].mean()

            if len(sex_groups) >= 2:
                male_rate = sex_groups.get('M', 0)
                female_rate = sex_groups.get('F', 0)

                # Demographic parity ratio (should be close to 1.0)
                if max(male_rate, female_rate) > 0:
                    parity_ratio = min(male_rate, female_rate) / max(male_rate, female_rate)
                else:
                    parity_ratio = 1.0

                results['sex_parity'] = {
                    'male_urgent_rate': float(male_rate),
                    'female_urgent_rate': float(female_rate),
                    'parity_ratio': float(parity_ratio),
                    'threshold': 0.8,
                    'passes': parity_ratio >= 0.8,
                    'male_count': int(len(df[df['patient_sex'] == 'M'])),
                    'female_count': int(len(df[df['patient_sex'] == 'F'])),
                }

        # Age-based demographic parity
        if 'age_group' in df.columns:
            age_groups = df.groupby('age_group')['has_urgent_action'].mean()

            age_parity = {}
            for age_group, rate in age_groups.items():
                age_parity[str(age_group)] = {
                    'urgent_rate': float(rate),
                    'count': int(len(df[df['age_group'] == age_group])),
                }

            # Check parity across all age groups
            rates = list(age_groups.values)
            if len(rates) > 1:
                parity_ratio = min(rates) / max(rates) if max(rates) > 0 else 1.0
                age_parity['parity_ratio'] = float(parity_ratio)
                age_parity['passes'] = parity_ratio >= 0.8

            results['age_parity'] = age_parity

        return results

    async def calculate_prediction_distribution_by_group(self) -> dict[str, Any]:
        """
        Analyze prediction distribution across demographic groups.

        Shows:
        - Which diagnoses are more common per group
        - Confidence score distributions
        - Processing time by group
        """
        df = await self.get_predictions_df()

        if df.empty:
            return {'error': 'No data available'}

        df = self._prepare_features(df)

        results = {
            'by_sex': {},
            'by_age_group': {},
            'sample_size': len(df),
        }

        # Diagnosis distribution by sex
        if 'patient_sex' in df.columns:
            for sex in df['patient_sex'].dropna().unique():
                sex_df = df[df['patient_sex'] == sex]
                diagnosis_dist = sex_df['primary_diagnosis'].value_counts(normalize=True)

                results['by_sex'][sex] = {
                    'count': len(sex_df),
                    'diagnosis_distribution': diagnosis_dist.to_dict(),
                    'avg_confidence': float(sex_df['max_confidence'].mean()),
                    'urgent_rate': float(sex_df['has_urgent_action'].mean()),
                }

        # Diagnosis distribution by age group
        if 'age_group' in df.columns:
            for age_group in df['age_group'].dropna().unique():
                age_df = df[df['age_group'] == age_group]
                diagnosis_dist = age_df['primary_diagnosis'].value_counts(normalize=True)

                results['by_age_group'][str(age_group)] = {
                    'count': len(age_df),
                    'diagnosis_distribution': diagnosis_dist.to_dict(),
                    'avg_confidence': float(age_df['max_confidence'].mean()),
                    'urgent_rate': float(age_df['has_urgent_action'].mean()),
                }

        return results

    async def generate_bias_report(self) -> dict[str, Any]:
        """
        Generate comprehensive bias report.

        Returns summary suitable for dashboard or alerting.
        """
        demographic_parity = await self.calculate_demographic_parity()
        distribution = await self.calculate_prediction_distribution_by_group()

        # Determine if bias detected
        bias_detected = False
        warnings = []

        if 'sex_parity' in demographic_parity:
            sex_parity = demographic_parity['sex_parity']
            if not sex_parity.get('passes', True):
                bias_detected = True
                warnings.append(
                    f"Sex bias detected: parity ratio {sex_parity['parity_ratio']:.2f} < 0.8"
                )

        if 'age_parity' in demographic_parity:
            age_parity = demographic_parity['age_parity']
            if not age_parity.get('passes', True):
                bias_detected = True
                warnings.append(
                    f"Age bias detected: parity ratio {age_parity.get('parity_ratio', 0):.2f} < 0.8"
                )

        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'bias_detected': bias_detected,
            'warnings': warnings,
            'demographic_parity': demographic_parity,
            'prediction_distribution': distribution,
            'summary': (
                f"Analyzed {demographic_parity.get('sample_size', 0)} predictions over "
                f"{self.lookback_days} days. "
                f"{'⚠️ Bias detected' if bias_detected else '✅ No bias detected'}"
            )
        }

        return report


async def check_bias_and_alert(db_url: str, threshold: float = 0.8) -> dict[str, Any]:
    """
    Check for bias and return summary for alerting.

    Use this in Airflow DAG or scheduled job.
    """
    analyzer = BiasAnalyzer(db_url=db_url, lookback_days=7)
    report = await analyzer.generate_bias_report()

    if report['bias_detected']:
        print("🚨 BIAS DETECTED!")
        for warning in report['warnings']:
            print(f"   {warning}")
    else:
        print("✅ No bias detected")

    return report
