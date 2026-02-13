#!/usr/bin/env python3
# ============================================================
# FILE: scripts/model_comparison_analysis.py
# ============================================================
"""
Generate comprehensive model comparison report from MLflow experiments.

Creates:
1. Performance comparison table
2. Improvement visualization
3. Trade-off analysis
4. Recommendation for production deployment

Usage:
    PYTHONPATH=. python scripts/model_comparison_analysis.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Output directory
output_dir = Path("reports/model_comparison")
output_dir.mkdir(parents=True, exist_ok=True)


def get_all_experiments():
    """Fetch all MLflow experiments."""

    experiments = [
        'ecg-baseline',
        'ecg-v2',
        'focal-loss-hyp-improvement',
        'fairness-constrained-training',
        'ensemble-models',
    ]

    all_runs = []

    for exp_name in experiments:
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp:
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["metrics.test_auc DESC"]
                )

                if not runs.empty:
                    runs['experiment'] = exp_name
                    all_runs.append(runs)
        except Exception as e:
            print(f"⚠️  Experiment '{exp_name}' not found: {e}")

    if not all_runs:
        print("❌ No experiments found!")
        return pd.DataFrame()

    combined = pd.concat(all_runs, ignore_index=True)

    # Deduplicate by run name - keep best AUC
    if 'tags.mlflow.runName' in combined.columns and 'metrics.test_auc' in combined.columns:
        combined = combined.sort_values('metrics.test_auc', ascending=False)
        combined = combined.drop_duplicates(subset=['tags.mlflow.runName'], keep='first')
        print(f"   Deduplicated to {len(combined)} unique runs")

    return combined


def create_comparison_table(df):
    """Create model comparison table."""

    columns_of_interest = [
        'tags.mlflow.runName',
        'experiment',
        'metrics.test_auc',
        'metrics.test_auc_HYP',
        'metrics.test_age_parity',
        'params.loss_function',
        'params.lambda_fairness',
        'params.focal_alpha',
    ]

    # Select available columns
    available_cols = [col for col in columns_of_interest if col in df.columns]
    comparison = df[available_cols].copy()

    # Rename for clarity
    comparison.columns = [col.split('.')[-1] for col in comparison.columns]

    # Sort by test_auc descending
    if 'test_auc' in comparison.columns:
        comparison = comparison.sort_values('test_auc', ascending=False)

    return comparison


def plot_auc_comparison(df, output_path):
    """Plot AUC comparison across models."""

    if 'metrics.test_auc' not in df.columns:
        print("⚠️  No test_auc metrics found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    models = df['tags.mlflow.runName'].fillna('Unknown').tolist()
    aucs = df['metrics.test_auc'].tolist()

    # Create bar plot
    bars = ax.bar(range(len(models)), aucs, color='steelblue', alpha=0.8)

    # Highlight best model
    best_idx = aucs.index(max(aucs))
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(1.0)

    # Add baseline line
    baseline_auc = 0.9318  # ResNet-18 v1
    ax.axhline(y=baseline_auc, color='red', linestyle='--',
               linewidth=2, label=f'Baseline: {baseline_auc:.4f}')

    # Formatting
    ax.set_ylabel('Test AUC', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0.85, 0.96)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Add values on bars
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.4f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ AUC comparison saved: {output_path}")
    plt.close()


def plot_hyp_improvement(df, output_path):
    """Plot HYP class improvement across models."""

    if 'metrics.test_auc_HYP' not in df.columns:
        print("⚠️  No HYP metrics found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    models = df['tags.mlflow.runName'].fillna('Unknown').tolist()
    hyp_aucs = df['metrics.test_auc_HYP'].fillna(0).tolist()

    bars = ax.bar(range(len(models)), hyp_aucs, color='coral', alpha=0.8)

    # Highlight models that achieve target
    for i, (bar, auc) in enumerate(zip(bars, hyp_aucs)):
        if auc > 0.90:
            bar.set_color('green')
            bar.set_alpha(1.0)

    # Target line
    ax.axhline(y=0.90, color='green', linestyle='--',
               linewidth=2, label='Target: 90%')

    # Baseline
    baseline_hyp = 0.8775
    ax.axhline(y=baseline_hyp, color='red', linestyle='--',
               linewidth=2, label=f'Baseline: {baseline_hyp:.4f}')

    ax.set_ylabel('HYP Class AUC', fontsize=12)
    ax.set_title('HYP (Hypertrophy) Performance Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0.85, 0.95)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Add values
    for bar, auc in zip(bars, hyp_aucs):
        if auc > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{auc:.4f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ HYP improvement saved: {output_path}")
    plt.close()


def plot_fairness_comparison(df, output_path):
    """Plot fairness metrics comparison."""

    if 'metrics.test_age_parity' not in df.columns:
        print("⚠️  No fairness metrics found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    models = df['tags.mlflow.runName'].fillna('Unknown')
    parity_ratios = df['metrics.test_age_parity'].fillna(0)

    bars = ax.bar(range(len(models)), parity_ratios, color='mediumpurple', alpha=0.8)

    # Highlight fair models
    for i, (bar, parity) in enumerate(zip(bars, parity_ratios)):
        if parity > 0.75:
            bar.set_color('green')
            bar.set_alpha(1.0)

    # Fairness threshold
    ax.axhline(y=0.80, color='green', linestyle='--',
               linewidth=2, label='Fair threshold: 0.80')

    # Baseline
    baseline_parity = 0.32
    ax.axhline(y=baseline_parity, color='red', linestyle='--',
               linewidth=2, label=f'Baseline: {baseline_parity}')

    ax.set_ylabel('Age Parity Ratio', fontsize=12)
    ax.set_title('Demographic Fairness Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Add values
    for bar, parity in zip(bars, parity_ratios):
        if parity > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{parity:.3f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Fairness comparison saved: {output_path}")
    plt.close()


def generate_summary_report(comparison_df):
    """Generate text summary report."""

    report = []
    report.append("=" * 70)
    report.append("MODEL COMPARISON SUMMARY")
    report.append("=" * 70)
    report.append("")

    # Best overall model
    if 'test_auc' in comparison_df.columns and not comparison_df.empty:
        best_model = comparison_df.iloc[0]
        report.append(f"🏆 Best Overall Model: {best_model.get('runName', 'Unknown')}")
        report.append(f"   Test AUC: {best_model['test_auc']:.4f}")
        report.append("")

    # Best for HYP
    if 'test_auc_HYP' in comparison_df.columns:
        hyp_data = comparison_df['test_auc_HYP'].dropna()
        if not hyp_data.empty:
            best_hyp_idx = hyp_data.idxmax()
            best_hyp = comparison_df.loc[best_hyp_idx]
            report.append(f"💊 Best for HYP (Hypertrophy): {best_hyp.get('runName', 'Unknown')}")
            report.append(f"   HYP AUC: {best_hyp['test_auc_HYP']:.4f}")
            report.append("")

    # Most fair model
    if 'test_age_parity' in comparison_df.columns:
        parity_data = comparison_df['test_age_parity'].dropna()
        if not parity_data.empty:
            fairest_idx = parity_data.idxmax()
            fairest = comparison_df.loc[fairest_idx]
            report.append(f"⚖️  Fairest Model: {fairest.get('runName', 'Unknown')}")
            report.append(f"   Age Parity: {fairest['test_age_parity']:.3f}")
            report.append("")

    # Production recommendation
    report.append("=" * 70)
    report.append("PRODUCTION RECOMMENDATION")
    report.append("=" * 70)
    report.append("")
    report.append("Consider trade-offs:")
    report.append("• Highest accuracy → Best overall AUC")
    report.append("• Clinical equity → Best fairness metrics")
    report.append("• Balanced approach → Ensemble model")
    report.append("")

    report_text = "\n".join(report)
    print(report_text)

    # Save to file
    report_path = output_dir / "comparison_summary.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\n📄 Summary saved: {report_path}")

    return report_text


def main():
    print("📊 Model Comparison Analysis")
    print(f"{'='*70}\n")

    # Get all experiments
    print("🔍 Fetching MLflow experiments...")
    df = get_all_experiments()

    if df.empty:
        print("❌ No experiments found in MLflow")
        print("   Make sure you've run training scripts first")
        return

    print(f"✅ Found {len(df)} runs across {df['experiment'].nunique()} experiments\n")

    # Create comparison table
    print("📋 Creating comparison table...")
    comparison = create_comparison_table(df)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(comparison.to_string(index=False))
    print()

    # Save table
    table_path = output_dir / "comparison_table.csv"
    comparison.to_csv(table_path, index=False)
    print(f"✅ Table saved: {table_path}\n")

    # Generate plots
    print("📈 Generating visualizations...")

    plot_auc_comparison(df, output_dir / "auc_comparison.png")
    plot_hyp_improvement(df, output_dir / "hyp_improvement.png")
    plot_fairness_comparison(df, output_dir / "fairness_comparison.png")

    print()

    # Generate summary report
    print("📝 Generating summary report...")
    summary = generate_summary_report(comparison)

    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)
    print(f"\nOutputs in: {output_dir}/")
    print("  • comparison_table.csv")
    print("  • auc_comparison.png")
    print("  • hyp_improvement.png")
    print("  • fairness_comparison.png")
    print("  • comparison_summary.txt")
    print()
    print("🎯 Next: Review results and select production model")


if __name__ == "__main__":
    main()
