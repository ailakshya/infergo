# infergo — AWS Test Plan

All hardware-blocked OPT tasks run here. Everything is scripted — run in order.

---

## Instances needed

| Script | Instance | GPUs | Cost/hr | OPTs tested |
|---|---|---|---|---|
| `provision.sh` | p3.8xlarge | 4× V100 16GB NVLink | ~$12 | OPT-23, OPT-24 |
| `provision.sh` | p3.2xlarge | 1× V100 32GB | ~$3 | P50 latency target |
| `provision.sh` | g4dn.xlarge × 2 | 1× T4 each | ~$0.53 each | OPT-26 |
| `run_opt25_eks.sh` | g4dn.xlarge × 3 (EKS) | 1× T4 each | ~$0.53 each + EKS | OPT-25 |

**Estimated total for one full test run: ~4–6 hours at <$100**

---

## Step 1 — AWS access

Two options:

### Option A: environment variables (simplest)
```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
```

### Option B: aws configure (interactive)
```bash
aws configure
# AWS Access Key ID: AKIA...
# AWS Secret Access Key: ...
# Default region: us-east-1
# Default output format: json
```

You also need an **EC2 key pair** for SSH access to instances:
```bash
# Create a key pair (saves .pem to local disk)
aws ec2 create-key-pair --key-name infergo-key \
  --query "KeyMaterial" --output text > ~/.ssh/infergo-key.pem
chmod 400 ~/.ssh/infergo-key.pem

export AWS_KEY_NAME=infergo-key
export KEY=~/.ssh/infergo-key.pem
```

---

## Step 2 — Provision instances

```bash
cd deploy/aws
chmod +x *.sh

AWS_KEY_NAME=infergo-key ./provision.sh
# Saves instance IDs to .aws_state
# Prints public IPs when ready
```

---

## Step 3 — Build infergo on each instance

Run this for the multi-gpu and p50 instances:
```bash
# Multi-GPU instance (OPT-23 + OPT-24)
ssh -i ~/.ssh/infergo-key.pem ubuntu@<MULTI_GPU_IP> 'bash -s' < setup_instance.sh

# P50 instance
ssh -i ~/.ssh/infergo-key.pem ubuntu@<P50_IP> 'bash -s' < setup_instance.sh

# Prefill + decode instances (OPT-26)
ssh -i ~/.ssh/infergo-key.pem ubuntu@<PREFILL_IP> 'bash -s' < setup_instance.sh
ssh -i ~/.ssh/infergo-key.pem ubuntu@<DECODE_IP>  'bash -s' < setup_instance.sh
```

---

## Step 4 — Download model on each instance

```bash
# On each instance:
ssh ubuntu@<IP> "$HOME/cgo/infergo pull \
  bartowski/Meta-Llama-3-8B-Instruct-GGUF \
  --quant Q4_K_M \
  --out \$HOME/models/llama3-8b-q4.gguf"
```

For OPT-23 T2 (70B model, needs p4d.24xlarge only):
```bash
ssh ubuntu@<MULTI_GPU_IP> "$HOME/cgo/infergo pull \
  bartowski/Meta-Llama-3-70B-Instruct-GGUF \
  --quant Q4_K_M \
  --out \$HOME/models/llama3-70b-q4.gguf"
```

---

## Step 5 — Run tests

```bash
# OPT-23 + OPT-24 (same instance, run sequentially)
ssh -i $KEY ubuntu@<MULTI_GPU_IP> 'bash -s' < run_opt23.sh
ssh -i $KEY ubuntu@<MULTI_GPU_IP> 'bash -s' < run_opt24.sh

# P50 latency target
ssh -i $KEY ubuntu@<P50_IP> 'bash -s' < run_p50.sh

# OPT-26 prefill/decode (orchestrated from local)
PREFILL_IP=<ip> DECODE_IP=<ip> KEY=$KEY ./run_opt26.sh

# OPT-25 EKS (run from local, takes ~30 min)
./run_opt25_eks.sh
```

---

## Step 6 — Collect results

```bash
MULTI_GPU_IP=<ip> P50_IP=<ip> KEY=$KEY ./collect_results.sh
# Pulls results to benchmarks/scalability/
# Claude then updates optimization_tasks.md + problemsolve.md and commits
```

---

## Step 7 — Teardown (stop billing)

```bash
./teardown.sh
# For EKS: eksctl delete cluster --name infergo-eks --region us-east-1
```

---

## IAM permissions needed

The AWS user/role needs:
- `ec2:*` (run, describe, terminate instances, manage security groups)
- `ecr:*` (push Docker image for OPT-25)
- `eks:*` (create/delete cluster for OPT-25)
- `iam:*` (OIDC for EKS node role — OPT-25 only)

Minimal policy for EC2-only tests (OPT-23/24/26/P50):
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["ec2:*"],
    "Resource": "*"
  }]
}
```
