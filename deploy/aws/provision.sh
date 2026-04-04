#!/usr/bin/env bash
# deploy/aws/provision.sh
# Provisions all AWS instances needed for OPT-23/24/25/26 and P50 tests.
# Run once. Saves instance IDs to .aws_state for use by other scripts.
#
# Usage: ./provision.sh [--region us-east-1] [--key-name my-key]
# Requires: aws CLI configured, a key pair in the target region.

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
KEY_NAME="${AWS_KEY_NAME:-infergo-key}"
STATE_FILE="$(dirname "$0")/.aws_state"

# Deep Learning AMI (Ubuntu 22.04) with CUDA 12.x — us-east-1
# Update AMI ID if using a different region: aws ec2 describe-images \
#   --owners amazon --filters "Name=name,Values=Deep Learning AMI GPU*Ubuntu*22.04*"
DLAMI_US_EAST_1="ami-0b6f627fa4af71e23"   # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3 (Ubuntu 22.04)

SG_NAME="infergo-sg"

echo "==> Region: $REGION | Key: $KEY_NAME"

# ── Security group ──────────────────────────────────────────────────────────
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" \
  --query "SecurityGroups[0].GroupId" --output text \
  --region "$REGION" 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
  echo "==> Creating security group $SG_NAME..."
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "infergo test instances" \
    --region "$REGION" \
    --query "GroupId" --output text)
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --region "$REGION" \
    --protocol tcp --port 22 --cidr 0.0.0.0/0
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --region "$REGION" \
    --protocol tcp --port 9090 --cidr 0.0.0.0/0
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --region "$REGION" \
    --protocol tcp --port 9091 --cidr 0.0.0.0/0
fi
echo "    SG: $SG_ID"

launch_instance() {
  local name="$1" type="$2" ami="$3"
  echo "==> Launching $name ($type)..."
  local iid
  iid=$(aws ec2 run-instances \
    --image-id "$ami" \
    --instance-type "$type" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$name},{Key=Project,Value=infergo}]" \
    --region "$REGION" \
    --query "Instances[0].InstanceId" --output text)
  echo "    $name -> $iid"
  echo "$name=$iid" >> "$STATE_FILE"
}

# Clear old state
> "$STATE_FILE"
echo "REGION=$REGION" >> "$STATE_FILE"
echo "KEY_NAME=$KEY_NAME" >> "$STATE_FILE"
echo "SG_ID=$SG_ID" >> "$STATE_FILE"
echo "AMI=$DLAMI_US_EAST_1" >> "$STATE_FILE"

# OPT-23 + OPT-24: 4× V100 (tensor-split + pipeline-stages)
launch_instance "infergo-multi-gpu"  "p3.8xlarge"   "$DLAMI_US_EAST_1"

# P50 latency: 1× V100 32GB (faster generation than RTX 5070 Ti)
launch_instance "infergo-p50"        "p3.2xlarge"   "$DLAMI_US_EAST_1"

# OPT-26 prefill node
launch_instance "infergo-prefill"    "g4dn.xlarge"  "$DLAMI_US_EAST_1"

# OPT-26 decode node
launch_instance "infergo-decode"     "g4dn.xlarge"  "$DLAMI_US_EAST_1"

echo ""
echo "==> Waiting for instances to reach running state..."
aws ec2 wait instance-running --region "$REGION" \
  --filters "Name=tag:Project,Values=infergo"

echo ""
echo "==> Public IPs:"
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=infergo" "Name=instance-state-name,Values=running" \
  --query "Reservations[].Instances[].[Tags[?Key=='Name'].Value|[0],PublicIpAddress]" \
  --output table --region "$REGION"

echo ""
echo "State saved to $STATE_FILE"
echo "Next: ./setup_instance.sh <ip> to build infergo on each instance"
