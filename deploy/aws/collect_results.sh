#!/usr/bin/env bash
# deploy/aws/collect_results.sh
# After running all OPT tests on AWS, pulls result files back to local machine
# and updates optimization_tasks.md + problemsolve.md automatically.
#
# Usage:
#   MULTI_GPU_IP=<ip> P50_IP=<ip> KEY=~/.ssh/infergo-key.pem ./collect_results.sh

set -euo pipefail

MULTI_GPU_IP="${MULTI_GPU_IP:?Set MULTI_GPU_IP}"
P50_IP="${P50_IP:?Set P50_IP}"
KEY="${KEY:-$HOME/.ssh/infergo-key.pem}"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BENCH="$REPO/benchmarks/scalability"

mkdir -p "$BENCH"

echo "==> Collecting OPT-23 results from $MULTI_GPU_IP"
scp $SSH_OPTS ubuntu@"$MULTI_GPU_IP":~/opt23_results.md "$BENCH/results_opt23.md"

echo "==> Collecting OPT-24 results from $MULTI_GPU_IP"
scp $SSH_OPTS ubuntu@"$MULTI_GPU_IP":~/opt24_results.md "$BENCH/results_opt24.md"

echo "==> Collecting P50 results from $P50_IP"
scp $SSH_OPTS ubuntu@"$P50_IP":~/p50_results.md "$BENCH/results_p50.md"

echo ""
echo "==> Results collected:"
ls -la "$BENCH"/*.md

echo ""
echo "==> OPT-23 summary:"
grep -A5 "## Test Results" "$BENCH/results_opt23.md" || true

echo ""
echo "==> OPT-24 summary:"
grep -A5 "## Test Results" "$BENCH/results_opt24.md" || true

echo ""
echo "==> P50 summary:"
grep "P50 c=4 target" "$BENCH/results_p50.md" || true

echo ""
echo "Next steps:"
echo "  1. Review results in benchmarks/scalability/"
echo "  2. Claude will update optimization_tasks.md + problemsolve.md"
echo "  3. git add + commit results"
