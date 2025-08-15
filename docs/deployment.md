# TechSafari 2K25 — End-to-End Deployment Guide (AWS EKS + Chaos)

This guide teaches a junior developer (no prior context on this app) how to deploy the entire Robo-Advisor platform to AWS using EKS and then run simple Chaos Engineering experiments. Follow each step carefully.

Note: You’ll incur AWS costs. Clean up when finished.

## 0) Prerequisites (install locally)

- AWS account with admin (or strong) permissions
- Tools installed on your machine:
  - AWS CLI v2: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
  - kubectl: https://kubernetes.io/docs/tasks/tools/
  - eksctl: https://eksctl.io/introduction/installation/
  - Helm 3: https://helm.sh/docs/intro/install/
  - Docker Desktop: https://www.docker.com/products/docker-desktop/
  - Node.js 18+ (optional for local checks)
- Configure AWS credentials:
  - Run `aws configure` and set your AWS Access Key, Secret, region (e.g., us-east-1), and output json

## 1) Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd robo-advisor-platform
```

## 2) Create Amazon ECR repositories (one per image)

Replace <ACCOUNT_ID> and <REGION> with your values.

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
REGISTRY="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY

for NAME in user-service market-data-service portfolio-service rebalancing-service llm-service trade-execution-service robo-frontend; do
  aws ecr create-repository --repository-name $NAME --region $REGION >/dev/null 2>&1 || true
  echo "Ensured ECR repo: $NAME"
done
```

## 3) Build and push Docker images

```bash
# Backend services
for SVC in user-service market-data-service portfolio-service rebalancing-service llm-service trade-execution-service; do
  docker build -t $SVC:latest ./services/$SVC
  docker tag $SVC:latest $REGISTRY/$SVC:latest
  docker push $REGISTRY/$SVC:latest
done

# Frontend
docker build -t robo-frontend:latest ./frontend
docker tag robo-frontend:latest $REGISTRY/robo-frontend:latest
docker push $REGISTRY/robo-frontend:latest
```

## 4) Create an EKS cluster (managed)

Option A (recommended): eksctl simple setup

```bash
CLUSTER_NAME=ts2k25-robo
eksctl create cluster \
  --name $CLUSTER_NAME \
  --region $REGION \
  --nodes 2 \
  --node-type t3.large \
  --managed
```

This takes ~15–25 minutes. When it completes, your kubeconfig context is configured automatically.

Verify access:
```bash
kubectl get nodes -o wide
```

## 5) Prepare Kubernetes namespace and data layer

```bash
# Create namespace, quotas, and basic limits
kubectl apply -f k8s/namespace.yaml

# Deploy Postgres and Redis (edit storageClassName in k8s/database.yaml to match your cluster, or remove it)
kubectl apply -f k8s/database.yaml

kubectl -n robo-advisor get pods,svc
```

Wait until the `postgres` and `redis` pods are Running.

## 6) App ConfigMaps and Secrets

Create ConfigMaps (shared URLs and frontend env):
```bash
kubectl apply -f infrastructure/k8s/configmaps/app-config.yaml
```

Create Secrets (base64-encode values, including your OpenAI key).

Option A — edit the file and apply:
- Open `infrastructure/k8s/secrets/app-secrets.yaml`
- Replace placeholders with base64-encoded values
  - Example: `echo -n 'your-jwt-secret' | base64`
- Apply:
```bash
kubectl apply -f infrastructure/k8s/secrets/app-secrets.yaml
```

Option B — create with kubectl (literals):
```bash
kubectl -n robo-advisor create secret generic app-secrets \
  --from-literal=JWT_SECRET='your-jwt-secret' \
  --from-literal=OPENAI_API_KEY='sk-...your key...' \
  --from-literal=PINECONE_API_KEY='optional' \
  --from-literal=PINECONE_ENVIRONMENT='optional' || true
```

## 7) Point Deployments to your ECR images

Edit these files under `infrastructure/k8s/deployments/` and set `image:` to your ECR URLs, e.g.:
- `image: <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/user-service:latest`
- Repeat for: market-data-service, portfolio-service, rebalancing-service, llm-service, trade-execution-service, frontend (robo-frontend)

Tip: You can also one-line patch via kubectl if you prefer.

## 8) Deploy application services

```bash
kubectl apply -f infrastructure/k8s/deployments/
```

Check status:
```bash
kubectl -n robo-advisor get pods -o wide
kubectl -n robo-advisor get svc
```

## 9) Access the frontend

The sample manifest exposes the frontend as a NodePort (30000).

- Get a node public IP:
```bash
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
echo "http://$NODE_IP:30000"
```
- If `ExternalIP` is empty (common on some setups), use the EC2 Console to find the node’s public IPv4 address.
- Alternatively, port-forward locally:
```bash
kubectl -n robo-advisor port-forward svc/frontend 3000:3000
# Open http://localhost:3000
```

Optional (Production): set up an ALB Ingress Controller and Ingress resource to expose the frontend over HTTPS with a domain.

## 10) Validate the app (health + smoke)

```bash
kubectl -n robo-advisor get pods
kubectl -n robo-advisor run tmp --rm -i --tty --image=curlimages/curl -- /bin/sh
# Inside the pod, try service DNS:
# curl http://user-service.robo-advisor.svc.cluster.local:8080/health
# curl http://rebalancing-service.robo-advisor.svc.cluster.local:8084/health
# curl -X POST http://rebalancing-service.robo-advisor.svc.cluster.local:8084/check-rebalance \
#   -H 'Content-Type: application/json' \
#   -d '{"portfolio":{"user_id":"demo","holdings":{"AAPL":10,"MSFT":10},"target_allocation":{"AAPL":0.6,"MSFT":0.4},"last_rebalanced":"2024-01-01T00:00:00Z"}}'
```

You can also run `./smoke-test.sh` against port-forwarded services.

## 11) Chaos Engineering (Chaos Mesh)

Install Chaos Mesh into the cluster (its own namespace):
```bash
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm repo update
kubectl create ns chaos-mesh || true
helm install chaos-mesh chaos-mesh/chaos-mesh -n chaos-mesh --set chaosDaemon.runtime=containerd --set chaosDaemon.socketPath=/run/containerd/containerd.sock
```

Verify:
```bash
kubectl -n chaos-mesh get pods
```

### Example experiments (apply in `robo-advisor` namespace)

1) Kill random pods in the app namespace (PodChaos):
```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: kill-random-pod
  namespace: robo-advisor
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces:
      - robo-advisor
  duration: '60s'
```
Apply:
```bash
kubectl apply -f kill-random-pod.yaml
```
Observe:
```bash
kubectl -n robo-advisor get pods
```

2) Add network delay to market-data-service (NetworkChaos):
```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: delay-market-data
  namespace: robo-advisor
spec:
  action: delay
  mode: all
  selector:
    labelSelectors:
      app: market-data-service
  delay:
    latency: '500ms'
    correlation: '25'
    jitter: '200ms'
  duration: '120s'
```
Apply and test UI responsiveness.

3) Stress CPU on llm-service (StressChaos):
```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: stress-llm
  namespace: robo-advisor
spec:
  mode: one
  selector:
    labelSelectors:
      app: llm-service
  stressors:
    cpu:
      workers: 2
      load: 80
  duration: '120s'
```

Clean up experiments:
```bash
kubectl -n robo-advisor delete podchaos --all
kubectl -n robo-advisor delete networkchaos --all
kubectl -n robo-advisor delete stresschaos --all
```

Tips:
- Watch the frontend health tiles and service `/health` endpoints during chaos.
- Re-run the README “Try the demo” steps to see resilience.

## 12) Scale and performance checks

Increase replicas (example: trade-execution-service to 3):
```bash
kubectl -n robo-advisor scale deploy/trade-execution-service --replicas=3
kubectl -n robo-advisor get pods -l app=trade-execution-service
```

Optional: install Metrics Server and configure HPAs later.

## 13) Cleanup

```bash
# App resources
kubectl delete ns robo-advisor || true
kubectl delete ns chaos-mesh || true

# EKS cluster
eksctl delete cluster --name $CLUSTER_NAME --region $REGION

# (Optional) Delete images
for NAME in user-service market-data-service portfolio-service rebalancing-service llm-service trade-execution-service robo-frontend; do
  aws ecr delete-repository --repository-name $NAME --region $REGION --force || true
done
```

## How this shows TechSafari 2K25 goals

- Scalability & Performance: microservices on EKS, easy replica scaling; add HPA later
- Resilience to Chaos: Chaos Mesh experiments (pod kill, latency, CPU stress) demonstrate robustness
- LLMs: Onboarding via llm-service (OpenAI API key in Secret) with timeouts/fallbacks
- ROI/Value: Rebalancing + trade execution produce tangible portfolio actions; extend with PnL dashboards

You now have a working cloud deployment and a chaos playbook to demo reliability under stress.
