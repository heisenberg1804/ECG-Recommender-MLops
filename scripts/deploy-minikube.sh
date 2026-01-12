#!/bin/bash
set -e

echo "🚀 Deploying ECG API to Minikube"
echo "================================"

# Check minikube
if ! minikube status > /dev/null 2>&1; then
    echo "❌ Minikube not running. Start with: minikube start"
    exit 1
fi

# Build in minikube
echo "🐳 Building image..."
eval $(minikube docker-env)
docker build -t ecg-api:v0.1.0 -f docker/Dockerfile.api .

# Deploy
echo "☸️  Deploying to Kubernetes..."
kubectl apply -k k8s/base/

# Wait
echo "⏳ Waiting for deployment..."
kubectl wait --for=condition=available --timeout=300s deployment/ecg-api -n ecg-dev

# Status
echo "✅ Deployment complete!"
kubectl get pods -n ecg-dev
echo ""
echo "Access with: kubectl port-forward -n ecg-dev svc/ecg-api 8000:80"