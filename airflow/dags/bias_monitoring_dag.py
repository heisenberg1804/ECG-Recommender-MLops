# ============================================================
# FILE: airflow/dags/bias_monitoring_dag.py
# ============================================================
"""
Bias monitoring DAG for ECG prediction system.

Runs: Daily
Purpose: Monitor fairness across demographic groups and alert via Email if bias is detected.
"""

import asyncio
import sys
from datetime import datetime, timedelta

from airflow.operators.email import EmailOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from airflow import DAG

# Add src to path
sys.path.insert(0, "/opt/airflow")

from src.monitoring.bias_analyzer import check_bias_and_alert


def run_bias_analysis(**context):
    """Run bias analysis and return boolean status (True if bias detected)."""
    # Database connection string
    db_url = "postgresql://ecg_user:ecg_password_dev@host.docker.internal:5432/ecg_predictions"

    # Run async analysis
    report = asyncio.run(check_bias_and_alert(db_url))

    # Push full report to XCom for downstream tasks
    context["ti"].xcom_push(key="bias_report", value=report)

    # Return boolean for the ShortCircuitOperator to check later
    return report["bias_detected"]


def generate_bias_summary(**context):
    """Generate human-readable summary in logs (Runs always)."""
    report = context["ti"].xcom_pull(task_ids='run_bias_analysis', key="bias_report")

    if not report:
        print("No report found.")
        return

    summary = []
    summary.append(f"Bias Analysis Report - {report['timestamp']}")
    summary.append(f"Sample size: {report['demographic_parity']['sample_size']}")
    summary.append("")

    if report["bias_detected"]:
        summary.append("🚨 BIAS DETECTED:")
        for warning in report["warnings"]:
            summary.append(f"  • {warning}")
    else:
        summary.append("✅ No bias detected - model is fair across demographics")

    # Sex parity details
    if "sex_parity" in report["demographic_parity"]:
        sex = report["demographic_parity"]["sex_parity"]
        summary.append("")
        summary.append("Sex Parity:")
        summary.append(f"  Male:   {sex['male_urgent_rate']:.1%} urgent (n={sex['male_count']})")
        summary.append(f"  Female: {sex['female_urgent_rate']:.1%} urgent (n={sex['female_count']})")
        summary.append(f"  Ratio:  {sex['parity_ratio']:.3f} ({'PASS' if sex['passes'] else 'FAIL'})")

    summary_text = "\n".join(summary)
    print(summary_text)


def log_bias_metrics(**context):
    """Log bias metrics for monitoring (Runs always)."""
    report = context["ti"].xcom_pull(task_ids='run_bias_analysis', key="bias_report")

    # Extract key metrics
    metrics = {}

    if "sex_parity" in report["demographic_parity"]:
        sex = report["demographic_parity"]["sex_parity"]
        metrics["sex_parity_ratio"] = sex["parity_ratio"]
        metrics["sex_parity_passes"] = 1 if sex["passes"] else 0

    if "age_parity" in report["demographic_parity"]:
        age = report["demographic_parity"]["age_parity"]
        metrics["age_parity_ratio"] = age.get("parity_ratio", 0)
        metrics["age_parity_passes"] = 1 if age.get("passes") else 0

    print(f"📊 Bias Metrics: {metrics}")


def check_if_bias_detected(**context):
    """
    ShortCircuitOperator function.
    Checks if bias was detected. Returns True to proceed (send email), False to stop.
    """
    # Pull the return value (boolean) from the analysis task
    bias_detected = context["ti"].xcom_pull(task_ids='run_bias_analysis')

    if bias_detected:
        print("🚨 Bias detected! Proceeding to alert sequence.")
        return True
    else:
        print("✅ No bias detected. Skipping email alert.")
        return False


def format_bias_email(**context):
    """Format email body for bias alert."""
    report = context['ti'].xcom_pull(task_ids='run_bias_analysis', key='bias_report')

    # Build HTML email
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .fail {{ color: #d32f2f; font-weight: bold; }}
            .pass {{ color: #388e3c; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h2>🚨 Bias Alert - ECG Clinical Recommender</h2>
        <p><strong>Timestamp:</strong> {report['timestamp']}</p>
        <p><strong>Sample Size:</strong> {report['demographic_parity']['sample_size']}</p>
        
        <h3>⚠️ Warnings:</h3>
        <ul>
        {''.join(f'<li>{w}</li>' for w in report['warnings'])}
        </ul>
    """

    if 'sex_parity' in report['demographic_parity']:
        sex = report['demographic_parity']['sex_parity']
        status_color = "pass" if sex['passes'] else "fail"
        status_text = "✅ PASS" if sex['passes'] else "❌ FAIL"

        html += f"""
        <h3>Sex-based Demographic Parity</h3>
        <table>
            <tr><th>Group</th><th>Urgent Rate</th><th>Count</th></tr>
            <tr><td>Male</td><td>{sex['male_urgent_rate']:.1%}</td><td>{sex['male_count']}</td></tr>
            <tr><td>Female</td><td>{sex['female_urgent_rate']:.1%}</td><td>{sex['female_count']}</td></tr>
        </table>
        <p><strong>Parity Ratio:</strong> {sex['parity_ratio']:.3f} (Threshold: 0.8)</p>
        <p><strong>Status:</strong> <span class="{status_color}">{status_text}</span></p>
        """

    html += """
        <br>
        <h3>Next Steps:</h3>
        <ol>
            <li>Review prediction logs for affected demographics</li>
            <li>Investigate model behavior on biased groups</li>
            <li>Consider retraining with fairness constraints</li>
        </ol>
    </body>
    </html>
    """

    return html


default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "ecg_bias_monitoring",
    default_args=default_args,
    description="Monitor fairness across demographic groups",
    schedule_interval="@daily",  # Run once per day
    start_date=datetime(2026, 1, 20),
    catchup=False,
    tags=["monitoring", "bias", "fairness", "ecg"],
) as dag:

    # 1. Run Analysis
    analyze_bias = PythonOperator(
        task_id="run_bias_analysis",
        python_callable=run_bias_analysis,
    )

    # 2. Generate Summary (Logs)
    create_summary = PythonOperator(
        task_id="generate_summary",
        python_callable=generate_bias_summary,
    )

    # 3. Log Metrics (Logs)
    log_metrics = PythonOperator(
        task_id="log_bias_metrics",
        python_callable=log_bias_metrics,
    )

    # 4. Check Condition (Circuit Breaker)
    # If False, downstream tasks (email) are skipped
    check_bias_condition = ShortCircuitOperator(
        task_id="check_if_bias_detected",
        python_callable=check_if_bias_detected,
    )

    # 5. Format Email Body (Only runs if bias detected)
    format_email = PythonOperator(
        task_id='format_email',
        python_callable=format_bias_email,
    )

    # 6. Send Email Alert (Only runs if bias detected)
    send_email_alert = EmailOperator(
        task_id='send_email_alert',
        to=['sahil.s.mohanty@gmail.com'],  # <--- UPDATE THIS EMAIL
        subject='🚨 Bias Detected - ECG Clinical Recommender',
        html_content="{{ ti.xcom_pull(task_ids='format_email') }}",
    )

    # Dependencies:
    # Run analysis -> Log Summary -> Log Metrics -> Check Condition -> Format -> Send
    analyze_bias >> create_summary >> log_metrics >> check_bias_condition >> format_email >> send_email_alert
