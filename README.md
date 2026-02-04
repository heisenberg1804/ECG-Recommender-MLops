# ECG Clinical Action Recommender - Production MLOps System

[![CI](https://github.com/heisenberg1804/ECG-Recommender-MLops/actions/workflows/ci.yml/badge.svg)](https://github.com/heisenberg1804/ECG-Recommender-MLops/actions/workflows/ci.yml)
[![CD](https://github.com/heisenberg1804/ECG-Recommender-MLops/actions/workflows/cd-staging.yml/badge.svg)](https://github.com/heisenberg1804/ECG-Recommender-MLops/actions/workflows/cd-staging.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end MLOps system that recommends clinical actions from 12-lead ECG signals, with production-grade monitoring, bias detection, and automated orchestration.

## 🎯 Project Overview

### What It Does

Given a 12-lead ECG signal and patient context (age, sex), the system:

1. **Analyzes the ECG** using a deep learning model (1D ResNet-18)
2. **Predicts diagnostic categories**: Normal, MI, ST/T Changes, Conduction Disturbance, Hypertrophy
3. **Recommends clinical actions** ranked by urgency and confidence
4. **Monitors for bias & drift** across demographic groups
5. **Alerts clinicians** when fairness or data quality issues are detected

### Why It Matters

- **Healthcare ML requires fairness** - Bias in recommendations can lead to disparate patient outcomes
- **Models drift over time** - Population changes, device upgrades, or data shifts degrade performance
- **Regulatory compliance** - Audit trails and explainability are mandatory for medical AI
- **Production readiness** - Shows complete MLOps lifecycle, not just model training

---

## 🚀 Live Demo
**Try the API here:** [http://35.224.1.181/docs](http://35.224.1.181/docs)

## 🏗️ Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ML SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │  12-lead ECG │ ───▶ │   ResNet-18  │ ───▶ │   Clinical   │  │
│  │   + Patient  │      │   Deep Model │      │    Actions   │  │
│  │   Context    │      │  (93% AUC)   │      │   Ranked     │  │
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
│  │  • Automated alerts (email notifications)                  │ │
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
- PTB-XL Dataset (21,799 ECG records)
- MLflow (experiment tracking & model registry)

**API & Serving:**
- FastAPI (REST API)
- Uvicorn (ASGI server)
- Pydantic (request/response validation)

**Infrastructure:**
- Docker (containerization)
- Kubernetes (orchestration)
- PostgreSQL (prediction logging)
- Redis (caching - planned)

**Monitoring & Observability:**
- Prometheus (metrics collection)
- Grafana (dashboards)
- Evidently AI (drift detection)
- Custom bias analyzer (fairness metrics)

**Orchestration:**
- Apache Airflow (workflow automation)
- Scheduled monitoring jobs
- Email alerting

**CI/CD:**
- GitHub Actions (automated testing & deployment)
- Git LFS (model versioning)
- GitHub Container Registry (image storage)

---

## 📊 Current Capabilities

### ✅ Implemented Features

#### Core ML Pipeline
- [x] Trained ResNet-18 model (93.18% test AUC)
- [x] Multi-label classification (5 diagnostic categories)
- [x] Clinical action mapping (diagnosis → recommendations)
- [x] Model versioning (v1, v2 with improvements)
- [x] MLflow experiment tracking

#### Production API
- [x] FastAPI serving with <15ms inference latency
- [x] Health checks (liveness + readiness probes)
- [x] Real-time ECG signal processing
- [x] Confidence-based recommendation ranking
- [x] Swagger documentation (/docs endpoint)

#### Monitoring & Governance
- [x] Prometheus metrics (predictions, latency, confidence)
- [x] **Drift detection** with Evidently AI
  - Input drift (patient demographics, signal characteristics)
  - Prediction drift (model output distribution changes)
  - Automated HTML reports
- [x] **Bias monitoring** with fairness analysis
  - Demographic parity (sex-based, age-based)
  - Prediction distribution by group
  - Alert thresholds (parity ratio < 0.8)
- [x] PostgreSQL prediction logging (full audit trail)
- [x] Grafana dashboards (system + custom metrics)

#### Orchestration & Automation
- [x] **Airflow DAGs** for monitoring automation
  - Hourly drift detection
  - Daily bias analysis
  - Email alerts when issues detected
- [x] Automated CI/CD pipeline
  - Lint & test on every push
  - Multi-architecture Docker builds (amd64/arm64)
  - Auto-push to GitHub Container Registry

#### Deployment
- [x] Docker containerization (2.2GB production image)
- [x] Kubernetes manifests (base + overlays for staging/prod)
- [x] ConfigMaps & Secrets management
- [x] Horizontal scaling ready (2 replicas)
- [x] Resource limits & requests defined

### 🔍 Discovered Issues (via Monitoring)

**Bias Detection Found:**
- Males receive urgent care recommendations 63.6% of the time
- Females receive urgent care recommendations 36.4% of the time
- **Parity ratio: 0.57** (fails 0.8 threshold)
- Root cause: Training data imbalance in MI diagnoses

**Drift Detection:**
- Prediction drift detected in test scenarios
- System correctly identified distribution shifts
- Would trigger model retraining in production

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

#### 4. Pull Model from Git LFS

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
./scripts/deploy-minikube.sh

# Start Airflow for monitoring
docker-compose -f docker/docker-compose.airflow.yml up -d
```

---

## 🧪 Testing the System

### Run API Tests

```bash
# Start API locally
uvicorn src.api.main:app --reload --port 8000

# In another terminal, run tests
PYTHONPATH=. python scripts/test_api.py
```

### Test with Real ECG

```bash
# The test script loads actual ECG signals from PTB-XL
# and sends them to the API, verifying:
# - Signal processing
# - Model inference
# - Action recommendation
# - Database logging
# - Metrics collection
```

### Monitor Bias & Drift

```bash
# Port-forward to PostgreSQL
kubectl port-forward -n ecg-dev svc/postgres 5432:5432

# Check for bias
PYTHONPATH=. python scripts/monitor_bias.py --check

# Check for drift
PYTHONPATH=. python scripts/monitor_drift.py --check

# Generate HTML drift report
PYTHONPATH=. python scripts/monitor_drift.py --report
open reports/drift/drift_report_*.html
```

### Access Monitoring Dashboards

- **Airflow:** http://localhost:8080 (admin/admin)
- **Grafana:** http://localhost:3000 (admin/[see installation logs])
- **Prometheus:** http://localhost:9090
- **MLflow:** `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`

---

## 📈 Model Training

### Train Baseline Model (v1)

```bash
PYTHONPATH=. python scripts/train.py \
  --data-dir ../data/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.001
```

**Results:**
- Test AUC: 93.18%
- Training time: ~2 hours
- Model size: 35MB

### Train Improved Model (v2)

```bash
PYTHONPATH=. python scripts/train_v2.py \
  --epochs 50 \
  --batch-size 64 \
  --dropout 0.3 \
  --patience 7 \
  --auto-register
```

**Improvements:**
- Early stopping (saves ~1 hour)
- Dropout regularization (reduces overfitting)
- Class weighting (addresses imbalance)
- Auto-registration to MLflow when AUC > 90%

**Results:**
- Test AUC: 92.77%
- Training time: ~1 hour (early stopped at epoch 17)
- Less overfitting than v1

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
  "ecg_id": "uuid",
  "diagnoses": [
    {"diagnosis": "MI", "confidence": 0.83}
  ],
  "recommendations": [
    {
      "action": "Activate cath lab",
      "confidence": 0.83,
      "urgency": "immediate",
      "reasoning": "Myocardial infarction detected - requires immediate intervention"
    }
  ],
  "model_version": "resnet18-v0.1.0",
  "processing_time_ms": 15.2
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

### Drift Detection

**Monitors:**
- Patient demographic distribution shifts
- Processing time changes
- Prediction confidence distribution
- Diagnosis category distribution

**Alerts when:**
- Statistical tests detect distribution changes (KS test, Chi-squared)
- Drift share exceeds threshold (>10% of features drifted)
- Prediction patterns deviate from baseline

**Runs:** Hourly via Airflow DAG

### Bias Monitoring

**Monitors:**
- Demographic parity across sex (male vs female)
- Demographic parity across age groups (<40, 40-65, >65)
- Urgent care recommendation rates by group
- Prediction confidence by demographic

**Alerts when:**
- Parity ratio < 0.8 (indicating disparate treatment)
- Recommendation rates differ significantly between groups

**Runs:** Daily via Airflow DAG

**Current Status:** System detected sex-based bias (parity ratio 0.57), triggering investigation

### Metrics Collected

**Business Metrics:**
- Total predictions by diagnosis type
- Recommendation urgency distribution
- Confidence score histograms

**Performance Metrics:**
- Prediction latency (p50, p95, p99)
- Request throughput
- Error rates

**ML Metrics:**
- Model loaded status
- Inference time breakdown
- Drift scores over time
- Fairness metrics by demographic

---

## 🔄 Automated Workflows (Airflow)

### DAG 1: Drift Monitoring (`ecg_drift_monitoring`)
**Schedule:** Every hour  
**Tasks:**
1. Detect drift (input + prediction)
2. Log drift metrics
3. Format alert email
4. Send email if drift detected

### DAG 2: Bias Monitoring (`ecg_bias_monitoring`)
**Schedule:** Daily  
**Tasks:**
1. Analyze demographic parity
2. Generate summary report
3. Log fairness metrics
4. Check bias thresholds
5. Format alert email
6. Send email if bias detected

---

## 🚢 Deployment

### Local Development (Minikube)

```bash
# Start minikube
minikube start

# Deploy application
kubectl apply -k k8s/base/

# Access API
kubectl port-forward -n ecg-dev svc/ecg-api 8000:80
```

### Production (Google Kubernetes Engine)

**Coming Soon:** Deployment to GKE with:
- Cloud SQL for PostgreSQL
- Cloud Monitoring integration
- Ingress with SSL
- Horizontal pod autoscaling

---

## 📁 Project Structure

```
ecg-recommender/
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI application
│   │   └── routes/
│   │       ├── predict.py          # Prediction endpoint
│   │       ├── health.py           # Health checks
│   │       ├── bias.py             # Bias monitoring API
│   │       └── drift.py            # Drift monitoring API
│   ├── ml/
│   │   ├── models/
│   │   │   └── resnet1d.py         # 1D ResNet architecture
│   │   ├── inference/
│   │   │   ├── predictor.py        # Inference logic
│   │   │   └── action_mapping.json # Diagnosis→Action mapping
│   │   └── preprocessing/
│   │       └── signal.py           # ECG signal processing
│   └── monitoring/
│       ├── drift_detector.py       # Evidently AI drift detection
│       └── bias_analyzer.py        # Fairness metrics calculation
├── scripts/
│   ├── train.py                    # Baseline model training
│   ├── train_v2.py                 # Improved model training
│   ├── monitor_drift.py            # CLI drift monitoring
│   ├── monitor_bias.py             # CLI bias monitoring
│   ├── test_api.py                 # Integration tests
│   └── deploy-minikube.sh          # Deployment automation
├── airflow/
│   ├── dags/
│   │   ├── drift_monitoring_dag.py # Drift monitoring workflow
│   │   └── bias_monitoring_dag.py  # Bias monitoring workflow
│   └── requirements.txt            # Airflow dependencies
├── k8s/
│   ├── base/                       # Base Kubernetes manifests
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── postgres.yaml
│   │   ├── mlflow.yaml
│   │   └── servicemonitor.yaml
│   └── overlays/
│       ├── staging/                # Staging environment
│       └── prod/                   # Production environment
├── docker/
│   ├── Dockerfile.api              # API container
│   └── docker-compose.airflow.yml  # Airflow stack
├── tests/
│   ├── unit/                       # Unit tests
│   └── integration/                # Integration tests
├── models/
│   └── best_model.pth              # Trained model (Git LFS)
├── pyproject.toml                  # Python dependencies
└── README.md
```

---

## 🎓 Key Learnings & Decisions

### 1. Model Versioning Strategy

**Chose:** Git LFS for production models + MLflow for experimentation

**Why:**
- Git LFS: Simple CI/CD integration, version controlled with code
- MLflow: Track 100s of experiments, hyperparameter search
- Best of both worlds: experiment with MLflow → promote best to Git LFS

### 2. Monitoring Architecture

**Chose:** Separate bias and drift monitoring

**Why:**
- Different cadences (hourly drift vs daily bias)
- Different stakeholders (engineers vs compliance teams)
- Different remediation strategies

### 3. Airflow vs K8s CronJobs

**Local:** Airflow for rich DAG features and debugging  
**Production:** Plan to use K8s CronJobs for simplicity and cost

**Why:**
- Airflow powerful but heavyweight (~$300/month for Cloud Composer)
- K8s CronJobs sufficient for scheduled monitoring
- Shows architectural maturity (choosing right tool for environment)

### 4. Bias Detection Findings

**Issue:** Model exhibits sex-based bias (males 1.75x more likely to get urgent recommendations)

**Hypothesis:** Training data (PTB-XL) has MI diagnosis skewed toward males

**Next Steps:**
1. Analyze training data for sex × diagnosis distribution
2. Apply fairness constraints in loss function
3. Collect more diverse data (CODE-15 Brazil, Chapman China)
4. Retrain with stratified sampling

---

## 🔮 Roadmap

### Phase 1: Core ML ✅ (COMPLETE)
- [x] Train baseline model
- [x] FastAPI serving
- [x] Docker + Kubernetes deployment
- [x] Basic monitoring

### Phase 2: Production Monitoring ✅ (COMPLETE)
- [x] Drift detection
- [x] Bias monitoring
- [x] Airflow orchestration
- [x] Email alerting

### Phase 3: Cloud Deployment 🔄 (IN PROGRESS)
- [ ] Deploy to Google Kubernetes Engine
- [ ] Cloud SQL for PostgreSQL
- [ ] Ingress with SSL certificate
- [ ] Cloud Monitoring integration
- [ ] Public demo URL

### Phase 4: Advanced Features 📋 (PLANNED)

**Multi-Dataset Integration:**
- [ ] Add CODE-15 dataset (345K Brazilian ECGs)
- [ ] Add Chapman dataset (45K Chinese ECGs)
- [ ] Geographic bias analysis
- [ ] Stratified sampling for global fairness

**Feature Engineering:**
- [ ] Extract clinical features (HRV, QT intervals, R-peaks)
- [ ] Feature store (Feast) for online/offline serving
- [ ] Combine deep + clinical features

**Model Improvements:**
- [ ] Explainability (Grad-CAM for ECG)
- [ ] Uncertainty quantification
- [ ] Cold-start handling for new patient types
- [ ] A/B testing framework (v1 vs v2)

**Advanced Monitoring:**
- [ ] Automated retraining pipeline
- [ ] Performance degradation alerts
- [ ] Calibration monitoring
- [ ] Slice-based analysis (performance by subgroup)

**Optimization:**
- [ ] ONNX export for cross-platform inference
- [ ] Model quantization (FP16/INT8)
- [ ] Batch inference optimization
- [ ] Redis caching for feature store

---

## 📝 Training Details

### Model Architecture

**1D ResNet-18** adapted for ECG signals:
- **Input:** (batch, 12, 5000) - 12 leads × 5000 samples at 500Hz
- **Encoder:** 4 residual blocks with increasing channels (64 → 128 → 256 → 512)
- **Pooling:** Adaptive average pooling
- **Output:** (batch, 5) - probabilities for 5 diagnostic classes
- **Parameters:** 8.7M trainable parameters

### Training Strategy

**Loss Function:** Binary Cross-Entropy with class weighting
```python
BCE_loss + λ₁ * fairness_penalty + λ₂ * L2_regularization
```

**Optimization:**
- Adam optimizer (lr=1e-3)
- ReduceLROnPlateau scheduler
- Early stopping (patience=7)

**Data Split:**
- Train: 72% (14,119 ECGs)
- Validation: 8% (1,569 ECGs)  
- Test: 20% (3,922 ECGs)

**Class Distribution:**
- NORM: 48.5%
- MI: 27.4%
- CD: 22.8%
- STTC: 14.1%
- HYP: 13.5% (smallest class)

### Evaluation Metrics

**Per-Class AUC (v1):**
- NORM: 96.10%
- MI: 94.22%
- STTC: 93.20%
- CD: 94.61%
- HYP: 87.75% ⚠️ (room for improvement)

**Macro AUC:** 93.18%

---

## 🔐 Security & Compliance

### Data Privacy
- No PHI (Protected Health Information) in logs
- Patient identifiers pseudonymized (UUIDs)
- Database access restricted by namespace

### Audit Trail
- Every prediction logged to PostgreSQL
- Immutable prediction records (UUID primary key)
- Timestamp, model version, patient demographics tracked

### Model Governance
- Model versioning in MLflow Model Registry
- Deployment approvals required (staging → production)
- Rollback capability via Kubernetes

---

## 🛠️ Development Workflow

### Local Development

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Make changes

# 3. Run tests
pytest tests/

# 4. Lint code
ruff check src/

# 5. Test locally
uvicorn src.api.main:app --reload

# 6. Commit and push
git add .
git commit -m "Add new feature"
git push origin feature/new-feature

# 7. Create PR → CI runs automatically
```

### Deploying New Model

```bash
# 1. Train model
python scripts/train_v2.py --auto-register

# 2. Model auto-registers to MLflow if AUC > 0.90

# 3. Copy model to Git LFS
cp models/best_model_v2.pth models/best_model.pth

# 4. Commit and push
git add models/best_model.pth
git commit -m "Deploy model v2"
git push

# 5. CI/CD automatically builds and deploys
```

---

## 📚 Resources & References

### Datasets
- [PTB-XL Database](https://physionet.org/content/ptb-xl/) - Primary training data
- [CODE-15% Database](https://physionet.org/content/code-15/) - Planned addition
- [Chapman-Shaoxing Database](https://physionet.org/content/chapman-shaoxing/) - Planned addition

### Papers & Guidelines
- Wagner et al., "PTB-XL: A Large Publicly Available ECG Dataset" (2020)
- AHA/ACC Clinical Guidelines for ECG Interpretation
- Mehrabi et al., "A Survey on Bias and Fairness in Machine Learning" (2021)

### Tools Documentation
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently AI Docs](https://docs.evidentlyai.com/)
- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)

---

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Sahil Mohanty**
- GitHub: [@heisenberg1804](https://github.com/heisenberg1804)
- Email: sahil.s.mohanty@gmail.com

---

## 🙏 Acknowledgments

- PTB-XL dataset creators (Physikalisch-Technische Bundesanstalt)
- Open-source ML/MLOps community
- FastAPI, PyTorch, and MLflow maintainers

---

## 📊 Project Stats

- **Lines of Code:** ~5,000+
- **Test Coverage:** 70%+
- **Docker Image Size:** 2.2GB
- **Model Size:** 35MB (8.7M parameters)
- **API Latency:** <15ms (p99)
- **Deployment:** Kubernetes (2 replicas, auto-scaling ready)
- **Monitoring:** 20+ Prometheus metrics tracked
- **Development Time:** 3 weeks

---

## 🚀 Quick Demo

Want to see it in action?

```bash
# 1. Clone and setup (5 min)
git clone https://github.com/heisenberg1804/ECG-Recommender-MLops.git
cd ECG-Recommender-MLops
git lfs pull
pip install -e .

# 2. Start services (2 min)
minikube start
./scripts/deploy-minikube.sh

# 3. Test prediction (30 sec)
kubectl port-forward -n ecg-dev svc/ecg-api 8000:80
PYTHONPATH=. python scripts/test_api.py

# 4. View monitoring (1 min)
# Airflow: http://localhost:8080
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

---

**⭐ If you find this project interesting, please star the repository!**

For questions or collaboration opportunities, feel free to reach out.
