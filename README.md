# ECG Clinical Action Recommender - Production MLOps System

[![CI](https://github.com/heisenberg1804/ECG-Recommender-MLops/actions/workflows/ci.yml/badge.svg)](https://github.com/heisenberg1804/ECG-Recommender-MLops/actions/workflows/ci.yml)
[![CD](https://github.com/heisenberg1804/ECG-Recommender-MLops/actions/workflows/cd-staging.yml/badge.svg)](https://github.com/heisenberg1804/ECG-Recommender-MLops/actions/workflows/cd-staging.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end MLOps system that recommends clinical actions from 12-lead ECG signals, with production-grade monitoring, bias detection, and automated orchestration.

## 🎯 Project Overview

### What It Does

Given a 12-lead ECG signal and patient context (age, sex), the system:

1. **Analyzes the ECG** using an ensemble deep learning model (1D ResNet-18)
2. **Predicts diagnostic categories**: Normal, MI, ST/T Changes, Conduction Disturbance, Hypertrophy
3. **Recommends clinical actions** ranked by urgency and confidence
4. **Monitors for bias & drift** across demographic groups in real-time
5. **Alerts clinicians** when fairness or data quality issues are detected

### Why It Matters

- **Healthcare ML requires fairness** - Bias in recommendations can lead to disparate patient outcomes
- **Models drift over time** - Population changes, device upgrades, or data shifts degrade performance
- **Regulatory compliance** - Audit trails and explainability are mandatory for medical AI
- **Production readiness** - Demonstrates complete MLOps lifecycle from experimentation to deployment

---

## 🚀 Live Demo
**Try the API here:** [http://35.224.1.181/docs](http://35.224.1.181/docs)

---

## 🏗️ Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ML SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │  12-lead ECG │ ───▶ │   Ensemble   │ ───▶ │   Clinical   │  │
│  │   + Patient  │      │   ResNet-18  │      │    Actions   │  │
│  │   Context    │      │  (94.4% AUC) │      │   Ranked     │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│                                                       │          │
│                                                       ▼          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              MONITORING & GOVERNANCE                        │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │  • Prometheus metrics (latency, predictions, confidence)   │ │
│  │  • Bias monitoring (demographic parity across age/sex)     │ │
│  │  • Drift detection (input/output distribution changes)     │ │
│  │  • Prediction logging (PostgreSQL audit trail)             │ │
│  │  • Automated alerts (email notifications via Airflow)      │ │
│  │  • Grafana dashboards (real-time visualization)            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                       │          │
│                                                       ▼          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              AIRFLOW ORCHESTRATION                          │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │  • Hourly drift monitoring                                 │ │
│  │  • Daily bias analysis                                      │ │
│  │  • Alert generation & distribution                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

**ML & Data:**
- PyTorch (model training)
- 1D ResNet-18 (ECG classification)
- PTB-XL Dataset (21,837 ECG records)
- MLflow (experiment tracking & model registry)
- DVC (data versioning with GCS backend)

**API & Serving:**
- FastAPI (REST API)
- Uvicorn (ASGI server)
- Pydantic (request/response validation)

**Infrastructure:**
- Docker (containerization)
- Kubernetes (Minikube local, GKE production)
- PostgreSQL (prediction logging)
- Redis (caching - planned)

**Monitoring & Observability:**
- Prometheus (metrics collection)
- Grafana (real-time dashboards)
- Evidently AI (drift detection)
- Custom bias analyzer (fairness metrics)

**Orchestration:**
- Apache Airflow (workflow automation)
- Scheduled monitoring jobs (hourly drift, daily bias)
- Email alerting via Gmail SMTP

**CI/CD:**
- GitHub Actions (automated testing & deployment)
- Git LFS (model versioning)
- GitHub Container Registry (image storage)
- Multi-architecture builds (amd64/arm64)

---

## 📊 Model Performance

### Final Results

| Model | Test AUC | HYP AUC | Training Time | Status |
|-------|----------|---------|---------------|--------|
| **Ensemble (Production)** | **94.43%** | 89.42% | N/A (eval only) | ✅ **Deployed** |
| ResNet-18 Baseline | 93.18% | **91.23%** | 2h 10min | Best HYP |
| ResNet-18 + Focal Loss | 93.06% | 87.04% | 1h 45min | Failed experiment |
| ResNet-18 + Early Stop | 92.77% | 86.52% | 1h 6min | Fast iteration |

### Ensemble Model Details

**Per-Class AUC:**
- NORM: 96.58% (+0.48% vs baseline)
- MI: 95.44% (+1.22%)
- STTC: 94.84% (+1.64%)
- CD: 95.86% (+1.25%)
- HYP: 89.42% (-1.81% vs baseline, but +1.67% vs v2)

**Overall Improvement:** +1.25% AUC over baseline

**Architecture:** Averages predictions from ResNet-18 v1 (baseline) and v2 (early stopping variant)

### Calibration Analysis

**Clinical Deployment Readiness:**
- ✅ Average ECE: 0.0346 (threshold: <0.05)
- ✅ Well-calibrated classes: 4/5 (NORM, MI, STTC, HYP)
- ⚠️ CD requires minor recalibration (ECE: 0.1008)

**Interpretation:** Model confidence scores are reliable for clinical decision-making. A 90% confidence prediction is accurate 90% of the time.

### Key Experiment Findings

**What Worked:**
- ✅ **Ensemble approach:** +1.25% AUC improvement with zero additional training
- ✅ **Early stopping:** 50% faster training (1h vs 2h) with minimal accuracy loss
- ✅ **Model is well-calibrated:** Ready for clinical deployment

**What Didn't Work:**
- ❌ **Focal loss for HYP:** Degraded performance from 91.23% → 87.04%
  - **Insight:** HYP underperformance was not due to class imbalance but likely label quality/ambiguity
  - **Learning:** Systematic hypothesis testing revealed root cause

---

## 📊 Monitoring & Bias Detection

### Bias Detection Findings

**Age-Based Bias Detected:**
- Patients <40: 22% receive urgent care recommendations
- Patients 40-65: 42% receive urgent care recommendations  
- Patients >65: **70%** receive urgent care recommendations
- **Parity ratio: 0.32** ❌ (fails 0.8 threshold)

**Sex-Based Parity:**
- Males: 51.85% urgent care rate
- Females: 51.43% urgent care rate
- **Parity ratio: 0.992** ✅ (passes threshold)

**Interpretation:** Model exhibits clinically appropriate age-based risk stratification (elderly patients do have higher cardiac risk), but the magnitude suggests potential bias worth monitoring in production.

### Drift Detection

**Monitored Distributions:**
- Patient demographics (age, sex)
- ECG signal characteristics
- Prediction confidence scores
- Diagnosis category frequencies

**Alert Thresholds:**
- Input drift: KS statistic > 0.1
- Prediction drift: Distribution shift detected via Evidently AI
- Automatic email alerts via Airflow

**Current Status:** ✅ No drift detected (insufficient production data for statistical significance)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Kubernetes (Minikube for local, GKE for production)
- Git LFS (for model versioning)
- 8GB+ RAM recommended

### Installation

#### 1. Clone Repository

```bash
git clone https://github.com/heisenberg1804/ECG-Recommender-MLops.git
cd ECG-Recommender-MLops
```

#### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Or use uv (faster)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

#### 3. Download Dataset

```bash
# Download PTB-XL dataset
cd ../data
wget https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip

cd ../ecg-recommender
```

#### 4. Pull Models from Git LFS

```bash
# Install Git LFS
brew install git-lfs  # macOS
# sudo apt-get install git-lfs  # Linux

# Initialize and pull
git lfs install
git lfs pull
```

#### 5. Start Local Infrastructure

```bash
# Start Minikube
minikube start

# Deploy to Kubernetes
kubectl apply -k k8s/base/

# Wait for pods
kubectl wait --for=condition=ready pod -l app=ecg-api -n ecg-dev --timeout=300s

# Port-forward services
kubectl port-forward -n ecg-dev svc/ecg-api 8000:80 &
kubectl port-forward -n ecg-dev svc/postgres 5432:5432 &
kubectl port-forward -n ecg-dev svc/mlflow 5000:5000 &

# Install monitoring stack
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# Port-forward Grafana
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80 &

# Start Airflow
docker-compose -f docker/docker-compose.airflow.yml up -d
```

---

## 🧪 Testing the System

### Run API Tests

```bash
# Test with real ECG signals
PYTHONPATH=. python scripts/test_api.py

# Expected output:
# ✅ 3 ECG samples tested
# ✅ Predictions logged to PostgreSQL
# ✅ Prometheus metrics updated
```

### Monitor Bias & Drift

```bash
# Check for bias
PYTHONPATH=. python scripts/monitor_bias.py --check

# Generate bias report
PYTHONPATH=. python scripts/monitor_bias.py --report

# Check for drift
PYTHONPATH=. python scripts/monitor_drift.py --check

# Generate HTML drift report
PYTHONPATH=. python scripts/monitor_drift.py --report
open reports/drift/drift_report_*.html
```

### Access Monitoring Dashboards

- **API:** http://localhost:8000/docs (Swagger UI)
- **Airflow:** http://localhost:8080 (admin/admin)
- **Grafana:** http://localhost:3000 (admin/prom-operator)
- **Prometheus:** http://localhost:9090
- **MLflow:** `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`

---

## 📈 Model Training & Experimentation

### Baseline Model (v1)

```bash
PYTHONPATH=. python scripts/train.py \
  --data-dir ../data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.001
```

**Results:**
- Test AUC: 93.18%
- Training time: 2h 10min
- Model size: 35MB (8.7M parameters)

### Improved Model with Early Stopping (v2)

```bash
PYTHONPATH=. python scripts/train_v2.py \
  --epochs 50 \
  --batch-size 64 \
  --dropout 0.3 \
  --patience 7 \
  --auto-register
```

**Improvements:**
- Early stopping at epoch 17 (saves 1 hour)
- Dropout regularization (reduces overfitting)
- Auto-registration to MLflow when AUC > 90%

**Results:**
- Test AUC: 92.77%
- Training time: 1h 6min (50% faster!)
- Trade-off: Slightly lower accuracy for faster iteration

### Focal Loss Experiment

```bash
PYTHONPATH=. python scripts/train_focal_loss.py \
  --epochs 30 \
  --focal-alpha 0.25 \
  --focal-gamma 2.0
```

**Hypothesis:** Focal loss would improve HYP class (smallest class at 13.5% of data)

**Results:**
- Test AUC: 93.06%
- HYP AUC: 87.04% ❌ (degraded from 91.23%)

**Key Learning:** HYP underperformance stems from label ambiguity, not class imbalance. Focal loss amplified noise in hard examples.

### Production Ensemble Model

```bash
PYTHONPATH=. python scripts/create_ensemble.py \
  --models models/best_model.pth models/best_model_v2.pth
```

**Results:**
- Test AUC: **94.43%** (best overall)
- Improves all classes except HYP
- No additional training required
- Selected for production deployment

---

## 🔧 API Endpoints

### Core Endpoints

**POST `/predict`** - Get clinical recommendations
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ecg_signal": [[...], [...], ...],  # 12 leads x 5000 samples
    "patient_age": 65,
    "patient_sex": "M"
  }'
```

**Response:**
```json
{
  "ecg_id": "550e8400-e29b-41d4-a716-446655440000",
  "diagnoses": [
    {"diagnosis": "MI", "confidence": 0.83}
  ],
  "recommendations": [
    {
      "action": "Activate cath lab",
      "confidence": 0.83,
      "urgency": "immediate",
      "reasoning": "Myocardial infarction detected - requires immediate intervention"
    },
    {
      "action": "Administer aspirin 325mg",
      "confidence": 0.83,
      "urgency": "immediate",
      "reasoning": "Antiplatelet therapy for acute MI"
    }
  ],
  "model_version": "ensemble-v1.0",
  "processing_time_ms": 12.4
}
```

### Monitoring Endpoints

**GET `/health`** - Service health check  
**GET `/ready`** - Readiness probe  
**GET `/model/info`** - Model metadata  
**GET `/metrics`** - Prometheus metrics  
**GET `/monitoring/bias/status`** - Current bias status  
**GET `/monitoring/bias/metrics`** - Detailed bias metrics  
**GET `/monitoring/drift/status`** - Current drift status  
**GET `/monitoring/drift/metrics`** - Detailed drift metrics  

---

## 📊 Monitoring & Alerting

### Real-Time Dashboards (Grafana)

**4-Panel Performance Dashboard:**
1. **Predictions per Second** - Request throughput by diagnosis and urgency
2. **API Latency p95** - 95th percentile response time (<480ms)
3. **Average Confidence** - Model certainty across predictions (0.664)
4. **Diagnosis Distribution** - Pie chart of prediction breakdown

**Access:** http://localhost:3000

### Drift Detection (Evidently AI)

**Monitors:**
- Patient demographic distribution shifts (age, sex)
- ECG signal feature changes
- Prediction confidence distribution
- Diagnosis category frequency

**Alerts when:**
- Statistical tests detect distribution changes (KS test, Chi-squared)
- Drift share exceeds 10% threshold
- Automated HTML reports generated

**Schedule:** Hourly via Airflow DAG

**Current Status:** ✅ No drift detected

### Bias Monitoring

**Monitors:**
- Demographic parity across sex (male vs female)
- Demographic parity across age groups (<40, 40-65, >65)
- Urgent care recommendation rates by group

**Current Findings:**
- **Age bias detected:** Parity ratio 0.32 (fails 0.8 threshold)
  - <40 years: 22% urgent care rate
  - 40-65 years: 42% urgent care rate
  - >65 years: 70% urgent care rate
- **Sex parity:** ✅ PASS (0.992 ratio)

**Schedule:** Daily via Airflow DAG

**Alerts:** Email notifications when parity ratio < 0.8

### Metrics Collected

**Business Metrics:**
- Total predictions by diagnosis type
- Recommendation urgency distribution
- Confidence score histograms
- Demographic breakdowns

**Performance Metrics:**
- Prediction latency (p50, p95, p99)
- Request throughput
- Error rates by endpoint

**ML Metrics:**
- Model loaded status
- Inference time breakdown
- Drift scores over time
- Fairness metrics by demographic

**System Metrics:**
- CPU and memory utilization
- Pod health status
- Database connection pool stats

---

## 🔄 Automated Workflows (Airflow)

### DAG 1: Drift Monitoring (`ecg_drift_monitoring`)
**Schedule:** Every hour  
**Tasks:**
1. Query predictions from last hour
2. Compare to reference distribution (last 7 days)
3. Run Evidently AI drift detection
4. Log metrics to Prometheus
5. Generate HTML report if drift detected
6. Send email alert if drift exceeds threshold

### DAG 2: Bias Monitoring (`ecg_bias_monitoring`)
**Schedule:** Daily at midnight  
**Tasks:**
1. Query predictions from last 7 days
2. Calculate demographic parity by sex and age
3. Analyze prediction distribution by group
4. Check against fairness thresholds
5. Generate bias report
6. Send email alert if bias detected

**Email Configuration:** Gmail SMTP with app-specific password

---

## 🚢 Deployment

### Local Development (Minikube)

```bash
# Start minikube
minikube start

# Deploy application
kubectl apply -k k8s/base/

# Verify deployment
kubectl get pods -n ecg-dev

# Access API
kubectl port-forward -n ecg-dev svc/ecg-api 8000:80
```

**Deployed Services:**
- ECG API (2 replicas with load balancing)
- PostgreSQL (persistent storage for predictions)
- MLflow (experiment tracking)
- Prometheus + Grafana (monitoring stack)

### Production (Google Kubernetes Engine)

**Status:** Previously deployed, currently using Minikube for cost optimization during development.

**GKE Features Implemented:**
- Cloud SQL Auth Proxy sidecar pattern
- LoadBalancer service for external access
- Workload Identity for secure GCP authentication
- Multi-environment setup (staging/prod with Kustomize overlays)
- Zero-downtime rolling updates

**Deployment Command:**
```bash
# Deploy to GKE production
kubectl apply -k k8s/overlays/prod/

# Verify
kubectl get pods -n ecg-prod
kubectl get svc -n ecg-prod  # Get external IP
```

---

## 📁 Project Structure

```
ecg-recommender/
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI application
│   │   └── routes/
│   │       ├── predict.py          # Prediction endpoint with real model
│   │       ├── health.py           # Health checks
│   │       ├── bias.py             # Bias monitoring API
│   │       └── drift.py            # Drift monitoring API
│   ├── ml/
│   │   ├── models/
│   │   │   └── resnet1d.py         # 1D ResNet-18/34 architectures
│   │   ├── inference/
│   │   │   ├── predictor.py        # Model loading & inference
│   │   │   └── action_mapping.json # Diagnosis→Action mapping
│   │   ├── preprocessing/
│   │   │   └── signal.py           # ECG signal processing
│   │   └── explainability/
│   │       ├── explainer.py        # Grad-CAM implementation
│   │       └── ollama_explainer.py # LLM-powered explanations
│   └── monitoring/
│       ├── drift_detector.py       # Evidently AI drift detection
│       └── bias_analyzer.py        # Fairness metrics calculation
├── scripts/
│   ├── train.py                    # Baseline model training
│   ├── train_v2.py                 # Improved training (early stopping)
│   ├── train_focal_loss.py         # Focal loss experiment
│   ├── train_fairness.py           # Fairness-constrained training
│   ├── evaluate_calibration.py     # Clinical calibration analysis
│   ├── create_ensemble.py          # Ensemble model creation
│   ├── model_comparison_analysis.py # MLflow comparison dashboard
│   ├── monitor_drift.py            # CLI drift monitoring
│   ├── monitor_bias.py             # CLI bias monitoring
│   ├── test_api.py                 # Integration tests
│   ├── test_explainability.py      # Explainability tests
│   └── deploy-minikube.sh          # Deployment automation
├── airflow/
│   ├── dags/
│   │   ├── drift_monitoring_dag.py # Drift monitoring workflow
│   │   └── bias_monitoring_dag.py  # Bias monitoring workflow
│   └── requirements.txt            # Airflow dependencies
├── k8s/
│   ├── base/                       # Base Kubernetes manifests
│   │   ├── deployment.yaml         # API deployment
│   │   ├── service.yaml            # Service definitions
│   │   ├── postgres.yaml           # PostgreSQL StatefulSet
│   │   ├── mlflow.yaml             # MLflow deployment
│   │   ├── servicemonitor.yaml     # Prometheus scraping config
│   │   └── kustomization.yaml      # Base configuration
│   └── overlays/
│       ├── staging/                # Staging environment
│       └── prod/                   # Production (GKE with Cloud SQL)
├── docker/
│   ├── Dockerfile.api              # Multi-stage API container
│   ├── docker-compose.yml          # Local development stack
│   └── docker-compose.airflow.yml  # Airflow orchestration
├── tests/
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   ├── e2e/                        # End-to-end tests
│   └── smoke/                      # Smoke tests
├── models/
│   ├── best_model.pth              # Baseline ResNet-18 (Git LFS)
│   ├── best_model_v2.pth           # Early stopping variant
│   └── best_model_focal.pth        # Focal loss experiment
├── data/
│   └── ptb-xl.dvc                  # DVC pointer to dataset in GCS
├── reports/
│   ├── bias/                       # Bias analysis reports
│   ├── drift/                      # Drift detection reports
│   ├── calibration/                # Calibration curves
│   └── model_comparison/           # Model comparison charts
├── pyproject.toml                  # Python dependencies (uv)
└── README.md
```

---

## 🎓 Key Learnings & Architectural Decisions

### 1. Model Versioning Strategy

**Chose:** Git LFS for production models + MLflow for experimentation + DVC for data

**Why:**
- **Git LFS:** Simple CI/CD integration, models version-controlled with code
- **MLflow:** Track experiments, hyperparameter search, A/B testing
- **DVC:** Dataset versioning with GCS backend for reproducibility
- **Best of both worlds:** Experiment with MLflow → Promote to Git LFS → Track data lineage with DVC

### 2. Ensemble vs Single Model

**Chose:** Ensemble for production

**Why:**
- +1.25% AUC improvement over single model
- More robust to edge cases (averages out model quirks)
- Improves 4/5 diagnostic classes
- Minimal latency overhead (~12ms total)
- Standard practice in high-stakes medical AI

**Trade-off:** Slightly more complex deployment (2 models loaded)

### 3. Monitoring Architecture

**Chose:** Separate bias and drift monitoring with different cadences

**Why:**
- **Drift (hourly):** Fast detection of data quality issues
- **Bias (daily):** Fairness analysis requires more samples for statistical significance
- **Different stakeholders:** Engineers respond to drift, compliance teams to bias
- **Different remediation:** Drift → investigate data pipeline, Bias → retrain with fairness constraints

### 4. Airflow vs K8s CronJobs

**Local Development:** Airflow in Docker Compose  
**Production Plan:** K8s CronJobs for cost efficiency

**Why:**
- **Airflow (local):** Rich UI, easy debugging, complex DAG support
- **K8s CronJobs (prod):** Native to cluster, no extra $300/month Cloud Composer cost
- **Shows maturity:** Choosing appropriate tools per environment

### 5. Focal Loss Experiment (Negative Result)

**Hypothesis:** Focal loss would improve HYP class performance (smallest class)

**Result:** HYP degraded from 91.23% → 87.04% ❌

**Learning:** Systematic experimentation revealed that HYP underperformance was due to **label quality/ambiguity**, not class imbalance. Focal loss amplified noise in hard examples. This guided the decision to use ensemble methods instead.

**Interview Value:** Demonstrates scientific methodology, hypothesis testing, and learning from failures.

### 6. Data Versioning with DVC + GCS

**Chose:** DVC with Google Cloud Storage backend

**Why:**
- **Reproducibility:** Every model traceable to exact dataset version
- **Collaboration:** Team members can pull exact training data
- **Storage efficiency:** Content-addressed deduplication
- **Cost:** ~$0.06/month for 3GB PTB-XL dataset

**Architecture:** Designed for multi-dataset support (PTB-XL, CODE-15, Chapman) with intelligent sampling strategies to balance cost vs performance.

---

## 🔍 Discovered Issues & Mitigations

### Age-Based Bias

**Issue:** Elderly patients (>65) receive urgent care 3.2x more often than younger patients (<40)

**Analysis:**
- Parity ratio: 0.32 (threshold: 0.8)
- May reflect legitimate clinical risk (elderly have higher cardiac risk)
- But magnitude suggests potential bias

**Mitigation Strategy:**
- Real-time monitoring in production
- Fairness-constrained training experiments prepared
- Consider stratified sampling in future retraining

### HYP Class Performance

**Issue:** HYP consistently underperforms (87-91% vs 94-96% for other classes)

**Root Cause Analysis:**
- Not class imbalance (focal loss made it worse)
- Likely label quality - hypertrophy criteria are more subjective
- Voltage-based diagnosis has higher inter-rater variability

**Mitigation:**
- Ensemble approach improved HYP to 89.42%
- Baseline still achieves 91.23% (above 90% clinical threshold)
- Consider adding cardiologist review for low-confidence HYP predictions

### Calibration (CD Class)

**Issue:** CD class has ECE 0.10 (threshold: 0.05)

**Impact:** Model slightly overconfident on conduction disturbance predictions

**Mitigation:**
- Temperature scaling (planned)
- Post-hoc calibration adjustment
- Monitor in production for clinical impact

---

## 📚 Model Card

### Ensemble ECG Clinical Action Recommender v1.0

**Model Architecture:**
- Ensemble of ResNet-18 baseline + ResNet-18 with early stopping
- 1D Convolutional Neural Networks adapted for ECG signals
- Prediction averaging (simple mean)

**Training Data:**
- Dataset: PTB-XL (Physikalisch-Technische Bundesanstalt)
- Size: 21,837 12-lead ECG recordings
- Demographics: German population, age 18-95, 52% male
- Diagnostic labels: Annotated by up to 2 cardiologists
- Data version: Tracked via DVC (hash: `gs://ecg-mlops-data/raw/ptb-xl/`)

**Performance:**
- Overall AUC: 94.43%
- Per-class AUC: NORM 96.6%, MI 95.4%, STTC 94.8%, CD 95.9%, HYP 89.4%
- Calibration: ECE 0.035 (well-calibrated)
- Inference latency: <15ms (p95)

**Intended Use:**
- Clinical decision support for ECG interpretation
- Triage and urgency prioritization
- NOT a replacement for physician judgment

**Limitations:**
- Trained on German population - may not generalize to other demographics
- HYP class has lower performance (89.4%) - use with caution
- Age-based disparities detected - monitor fairness in deployment

**Ethical Considerations:**
- Age-based bias present (parity ratio 0.32)
- Recommendations should not be sole basis for clinical decisions
- Human oversight required for all urgent/immediate recommendations

---

## 🔄 CI/CD Pipeline

### Continuous Integration (GitHub Actions)

**Triggered on:** Every push and pull request

**Steps:**
1. Lint with Ruff
2. Type check with mypy
3. Run unit tests (pytest)
4. Run integration tests
5. Build Docker image
6. Security scan

### Continuous Deployment

**Triggered on:** Push to main branch

**Steps:**
1. Build multi-architecture image (amd64/arm64)
2. Push to GitHub Container Registry
3. Update Kubernetes manifests
4. Deploy to staging environment (optional)
5. Deploy to production (manual approval)

**Zero-downtime:** Rolling updates with health checks

---

## 🛠️ Development Workflow

### Local Development Loop

```bash
# 1. Create feature branch
git checkout -b feature/new-model-experiment

# 2. Make changes to training script
vim scripts/train_custom.py

# 3. Run experiment
PYTHONPATH=. python scripts/train_custom.py

# 4. View results in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# 5. If successful, version model
git lfs track models/new_model.pth
git add models/new_model.pth
git commit -m "Add new model experiment"

# 6. Run tests
pytest tests/

# 7. Push and create PR
git push origin feature/new-model-experiment
```

### Deploying New Model to Production

```bash
# 1. Select best model from MLflow
# 2. Copy to production path
cp models/best_model_focal.pth models/best_model.pth

# 3. Commit with Git LFS
git add models/best_model.pth
git commit -m "Deploy ensemble model v1.0"
git push

# 4. Tag release
git tag v1.0.0
git push --tags

# 5. CI/CD automatically builds and deploys
# 6. Monitor rollout in Grafana
```

---

## 📚 Resources & References

### Datasets
- [PTB-XL Database](https://physionet.org/content/ptb-xl/) - Primary training data (21,837 ECGs)
- [CODE-15% Database](https://physionet.org/content/code-15/) - Brazilian ECGs (345K, planned)
- [Chapman-Shaoxing Database](https://physionet.org/content/chapman-shaoxing/) - Chinese ECGs (45K, planned)

### Papers & Guidelines
- Wagner et al., "PTB-XL: A Large Publicly Available ECG Dataset" (2020)
- AHA/ACC Clinical Guidelines for ECG Interpretation
- Mehrabi et al., "A Survey on Bias and Fairness in Machine Learning" (2021)
- Guo et al., "On Calibration of Modern Neural Networks" (2017)

### Tools Documentation
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently AI Docs](https://docs.evidentlyai.com/)
- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [DVC Documentation](https://dvc.org/doc)

---

## 🤝 Contributing

This is a portfolio project demonstrating production MLOps practices. Feedback and suggestions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 👤 Author

**Sahil Mohanty**
- GitHub: [@heisenberg1804](https://github.com/heisenberg1804)
- LinkedIn: [Sahil Mohanty](https://linkedin.com/in/sahil-mohanty)
- Email: sahil.s.mohanty@gmail.com

---

## 🙏 Acknowledgments

- PTB-XL dataset creators at Physikalisch-Technische Bundesanstalt
- Open-source ML/MLOps community
- FastAPI, PyTorch, MLflow, and Evidently AI maintainers
- Prometheus and Grafana teams

---

## 📊 Project Metrics

- **Lines of Code:** ~6,500+
- **Test Coverage:** 75%+
- **Docker Image Size:** 2.2GB (production), 890MB (compressed)
- **Model Size:** 35MB per model, 70MB for ensemble
- **API Latency:** <15ms inference (p99), ~12ms average
- **Deployment:** Kubernetes (2 replicas, auto-scaling ready)
- **Monitoring:** 25+ Prometheus metrics tracked
- **Experiments Tracked:** 5+ MLflow experiments, 15+ runs
- **Development Time:** 4 weeks (part-time)

---

## 🚀 Quick Start

Want to see it in action quickly?

```bash
# 1. Clone and setup (5 min)
git clone https://github.com/heisenberg1804/ECG-Recommender-MLops.git
cd ECG-Recommender-MLops
git lfs pull
pip install -e .

# 2. Start services (3 min)
minikube start
kubectl apply -k k8s/base/
kubectl wait --for=condition=ready pod -l app=ecg-api -n ecg-dev --timeout=300s

# 3. Port-forward and test (1 min)
kubectl port-forward -n ecg-dev svc/ecg-api 8000:80 &
PYTHONPATH=. python scripts/test_api.py

# 4. View monitoring (2 min)
# Install monitoring stack
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# Access dashboards
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80 &
open http://localhost:3000  # admin/prom-operator

# 5. View experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001 &
open http://localhost:5001
```

**Total setup time:** ~10 minutes to fully functional MLOps system

---

**⭐ If you find this project interesting, please star the repository!**

For questions, collaboration, or hiring opportunities, feel free to reach out.