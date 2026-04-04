#!/usr/bin/env bash
# deploy/aws/teardown.sh
# Terminates all infergo AWS instances to stop billing.
# Reads instance IDs from .aws_state written by provision.sh.
#
# Usage: ./teardown.sh

set -euo pipefail

STATE_FILE="$(dirname "$0")/.aws_state"

if [[ ! -f "$STATE_FILE" ]]; then
  echo "No .aws_state file found. Nothing to tear down."
  exit 0
fi

source "$STATE_FILE"

echo "==> Terminating infergo instances in $REGION"

IDS=()
while IFS='=' read -r key val; do
  case "$key" in
    infergo-*) IDS+=("$val") ;;
  esac
done < "$STATE_FILE"

if [[ ${#IDS[@]} -eq 0 ]]; then
  echo "No instance IDs found in $STATE_FILE"
  exit 0
fi

echo "    Instances: ${IDS[*]}"
read -r -p "Terminate these instances? [y/N] " confirm
[[ "$confirm" != "y" ]] && { echo "Aborted."; exit 0; }

aws ec2 terminate-instances \
  --instance-ids "${IDS[@]}" \
  --region "$REGION" \
  --query "TerminatingInstances[].{ID:InstanceId,State:CurrentState.Name}" \
  --output table

echo ""
echo "==> Instances terminating. Cost stops within ~1 minute."
echo "    To delete the security group after instances terminate:"
echo "    aws ec2 delete-security-group --group-id $SG_ID --region $REGION"

rm -f "$STATE_FILE"
echo "    .aws_state cleared."
