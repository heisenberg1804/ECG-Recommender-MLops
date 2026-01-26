"""
Drift monitoring DAG for ECG prediction system.

Runs: Every hour
Purpose: Detect input/output drift and alert via Email if detected.
"""

import asyncio
import sys
from datetime import datetime, timedelta

from airflow.operators.email import EmailOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from airflow import DAG

# Add src to path
sys.path.insert(0, "/opt/airflow")

# Ensure this module exists in your src folder, or the DAG will fail to import
from src.monitoring.drift_detector import check_drift_and_alert


def run_drift_detection(**context):
    """Run drift detection and return results."""
    # Database connection string (matches your docker-compose service names)
    db_url = "postgresql://ecg_user:ecg_password_dev@host.docker.internal:5432/ecg_predictions"

    # Run async drift detection
    summary = asyncio.run(check_drift_and_alert(db_url, threshold=0.1))

    # Push to XCom for downstream tasks
    context["ti"].xcom_push(key="drift_summary", value=summary)

    return summary


def log_drift_metrics(**context):
    """Log drift metrics to monitoring system (Runs always)."""
    summary = context["ti"].xcom_pull(task_ids='detect_drift', key="drift_summary")

    if not summary:
        print("No summary found in XCom.")
        return

    print("📊 Drift Metrics:")
    print(f"   Input drift: {summary['input_drift'].get('drift_detected', False)}")
    print(f"   Prediction drift: {summary['prediction_drift'].get('drift_detected', False)}")
    print(f"   Timestamp: {summary['timestamp']}")


def check_if_alert_needed(**context):
    """
    ShortCircuitOperator function. 
    Returns True to proceed (send email), False to skip downstream tasks.
    """
    summary = context["ti"].xcom_pull(task_ids='detect_drift', key="drift_summary")

    # Check the boolean 'alert' key from your drift detector
    should_alert = summary.get("alert", False)

    if should_alert:
        print("🚨 Drift detected! Proceeding to send email.")
        return True
    else:
        print("✅ No drift detected. Skipping email alert.")
        return False


def format_drift_email(**context):
    """Format email body for drift alert."""
    summary = context['ti'].xcom_pull(task_ids='detect_drift', key='drift_summary')

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .alert {{ color: #d32f2f; font-weight: bold; }}
            .metric {{ background-color: #f5f5f5; padding: 10px; margin-bottom: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h2>🚨 Drift Alert - ECG Clinical Recommender</h2>
        <p><strong>Timestamp:</strong> {summary['timestamp']}</p>

        <div class="metric">
            <h3>Input Drift:</h3>
            <p>Detected: <span class="alert">{summary['input_drift'].get('drift_detected', False)}</span></p>
            <p>Drift Share: <strong>{summary['input_drift'].get('drift_share', 0):.2%}</strong></p>
        </div>

        <div class="metric">
            <h3>Prediction Drift:</h3>
            <p>Detected: <span class="alert">{summary['prediction_drift'].get('drift_detected', False)}</span></p>
        </div>

        <h3>Action Required:</h3>
        <ol>
            <li>Review drift report in Grafana</li>
            <li>Investigate root cause of distribution shift</li>
            <li>Consider model retraining if drift persists</li>
        </ol>
    </body>
    </html>
    """

    # We return the HTML string so the EmailOperator can pull it via XCom return_value
    return html


default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "ecg_drift_monitoring",
    default_args=default_args,
    description="Monitor for drift in ECG predictions",
    schedule_interval="@hourly",
    start_date=datetime(2026, 1, 20),
    catchup=False,
    tags=["monitoring", "drift", "ecg"],
) as dag:

    # 1. Detect Drift
    detect_drift = PythonOperator(
        task_id="detect_drift",
        python_callable=run_drift_detection,
    )

    # 2. Log Metrics (Always runs after detection)
    log_metrics = PythonOperator(
        task_id="log_drift_metrics",
        python_callable=log_drift_metrics,
    )

    # 3. Check Condition (Circuit Breaker)
    # If this returns False, tasks downstream (format & send email) are skipped
    check_alert_condition = ShortCircuitOperator(
        task_id="check_if_alert_needed",
        python_callable=check_if_alert_needed,
    )

    # 4. Format Email (Only if drift detected)
    format_email = PythonOperator(
        task_id='format_email',
        python_callable=format_drift_email,
    )

    # 5. Send Email (Only if drift detected)
    # Uses the Global SMTP config from docker-compose
    send_email_alert = EmailOperator(
        task_id='send_email_alert',
        to=['sahil.s.mohanty@gmail.com'],  # <--- UPDATE THIS EMAIL
        subject='🚨 Drift Detected - ECG Clinical Recommender',
        # Pulls return value from format_email task
        html_content="{{ ti.xcom_pull(task_ids='format_email') }}",
    )

    # Define Dependencies
    # detect -> log -> check -> format -> send
    detect_drift >> log_metrics >> check_alert_condition >> format_email >> send_email_alert
