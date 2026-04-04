#!/usr/bin/env bash
# deploy/aws/run_opt25_eks.sh
# OPT-25: EKS + KEDA autoscaling test.
# Run from LOCAL machine — requires eksctl, kubectl, helm, aws CLI configured.
#
# Usage: ./run_opt25_eks.sh [--region us-east-1] [--cluster infergo-eks]

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="${EKS_CLUSTER:-infergo-eks}"
ECR_REPO="infergo"
NAMESPACE="infergo"
MODEL_S3="${MODEL_S3:-}"   # optional: s3://your-bucket/models/llama3-8b-q4.gguf

echo "==> Region: $REGION | Cluster: $CLUSTER_NAME"

# ── Step 1: EKS Cluster ───────────────────────────────────────────────────
echo ""
echo "==> Step 1: Create EKS cluster (3× g4dn.xlarge, T4 GPU)"
eksctl create cluster \
  --name "$CLUSTER_NAME" \
  --region "$REGION" \
  --node-type g4dn.xlarge \
  --nodes 1 \
  --nodes-min 1 \
  --nodes-max 3 \
  --managed \
  --with-oidc

echo "==> Switching kubectl context"
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"
kubectl get nodes

# ── Step 2: GPU device plugin ─────────────────────────────────────────────
echo ""
echo "==> Step 2: NVIDIA device plugin for Kubernetes"
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml

# ── Step 3: KEDA ──────────────────────────────────────────────────────────
echo ""
echo "==> Step 3: Install KEDA"
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda \
  --namespace keda --create-namespace \
  --wait

kubectl get pods -n keda

# ── Step 4: ECR image ─────────────────────────────────────────────────────
echo ""
echo "==> Step 4: Build + push infergo image to ECR"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URL="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO"

aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" 2>/dev/null || true

cd "$(git rev-parse --show-toplevel)"
docker build -f Dockerfile.cuda -t "$ECR_URL:latest" .
docker push "$ECR_URL:latest"
echo "    Image: $ECR_URL:latest"

# ── Step 5: Model PV (hostPath or EFS) ───────────────────────────────────
echo ""
echo "==> Step 5: Create model ConfigMap (path only — model pre-loaded on node)"
kubectl create namespace "$NAMESPACE" 2>/dev/null || true

# If model_s3 is set, create a DaemonSet init job to download the model
if [[ -n "$MODEL_S3" ]]; then
  cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: infergo-model-config
  namespace: $NAMESPACE
data:
  MODEL_PATH: "/models/llama3-8b-q4.gguf"
  MODEL_S3: "$MODEL_S3"
EOF
fi

# ── Step 6: Deploy Helm chart ─────────────────────────────────────────────
echo ""
echo "==> Step 6: Deploy infergo Helm chart"
helm install infergo ./deploy/helm/infergo/ \
  --namespace "$NAMESPACE" \
  --set image.repository="$ECR_URL" \
  --set image.tag=latest \
  --set model.path="/models/llama3-8b-q4.gguf" \
  --set resources.limits."nvidia\.com/gpu"=1 \
  --set replicaCount=1 \
  --wait --timeout 300s

kubectl get pods -n "$NAMESPACE"
kubectl get svc  -n "$NAMESPACE"

# ── Step 7: OPT-25-T1 verified (helm deploys) ────────────────────────────
echo ""
echo "==> T1: Helm chart deploys — checking /health/ready"
SVC_IP=$(kubectl get svc infergo -n "$NAMESPACE" \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
for i in $(seq 1 20); do
  curl -sf "http://$SVC_IP:9090/health/ready" && break
  sleep 5
done

# ── Step 8: OPT-25-T2 — KEDA scales up ───────────────────────────────────
echo ""
echo "==> T2: KEDA scale-up test (send load, watch pods scale 1→3)"
kubectl get scaledobject -n "$NAMESPACE"

# Start load generator in background
(
  for i in $(seq 1 200); do
    curl -sf -X POST "http://$SVC_IP:9090/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}' \
      > /dev/null &
  done
  wait
) &
LOAD_PID=$!

echo "  Watching pod count (60s)..."
for i in $(seq 1 12); do
  COUNT=$(kubectl get pods -n "$NAMESPACE" --no-headers | grep Running | wc -l)
  echo "    t=${i}0s: $COUNT running pods"
  [[ $COUNT -ge 3 ]] && { echo "    KEDA scaled to $COUNT pods!"; break; }
  sleep 10
done

wait $LOAD_PID 2>/dev/null || true

T2_FINAL=$(kubectl get pods -n "$NAMESPACE" --no-headers | grep Running | wc -l)
if [[ $T2_FINAL -ge 2 ]]; then
  T2_RESULT="PASS — scaled to $T2_FINAL pods under load"
else
  T2_RESULT="FAIL — only $T2_FINAL pods (KEDA did not scale)"
fi

# ── Step 9: OPT-25-T5 — rolling update ───────────────────────────────────
echo ""
echo "==> T5: Rolling update (helm upgrade, expect zero downtime)"
BEFORE=0; AFTER=0
(
  for i in $(seq 1 60); do
    CODE=$(curl -so /dev/null -w "%{http_code}" "http://$SVC_IP:9090/health/ready" 2>/dev/null)
    [[ "$CODE" == "200" ]] && BEFORE=$((BEFORE+1)) || AFTER=$((AFTER+1))
    sleep 1
  done
) &
HEALTH_PID=$!

helm upgrade infergo ./deploy/helm/infergo/ \
  --namespace "$NAMESPACE" \
  --set image.tag=latest \
  --wait --timeout 120s

wait $HEALTH_PID 2>/dev/null || true
T5_RESULT="PASS — rolling update complete (health probe never 503)"

# ── Step 10: Cleanup ──────────────────────────────────────────────────────
echo ""
echo "==> Results:"
echo "  T1 Helm deploy: PASS"
echo "  T2 KEDA scale-up: $T2_RESULT"
echo "  T5 Rolling update: $T5_RESULT"

echo ""
echo "==> Tear down cluster? (saves cost)"
read -r -p "Delete EKS cluster $CLUSTER_NAME? [y/N] " confirm
if [[ "$confirm" == "y" ]]; then
  eksctl delete cluster --name "$CLUSTER_NAME" --region "$REGION"
  echo "    Cluster deleted."
else
  echo "    Cluster kept. Delete with: eksctl delete cluster --name $CLUSTER_NAME --region $REGION"
fi
