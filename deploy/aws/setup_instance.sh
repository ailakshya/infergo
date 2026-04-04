#!/usr/bin/env bash
# deploy/aws/setup_instance.sh
# Runs ON a fresh Deep Learning AMI instance.
# Installs dependencies, clones repo, builds infergo binary + shared lib.
#
# Usage (from local machine):
#   ssh -i ~/.ssh/infergo-key.pem ubuntu@<ip> 'bash -s' < setup_instance.sh
# Or copy to instance and run directly.

set -euo pipefail

REPO_URL="https://github.com/ailakshya/infergo.git"
BRANCH="main"
BUILD_DIR="$HOME/cgo"
MODEL_DIR="$HOME/models"

echo "==> System packages"
sudo apt-get update -q
sudo apt-get install -y -q \
  git cmake ninja-build build-essential \
  libopenblas-dev pkg-config \
  wget curl jq

echo "==> Go 1.22"
if ! command -v go &>/dev/null; then
  wget -q https://go.dev/dl/go1.22.4.linux-amd64.tar.gz
  sudo tar -C /usr/local -xzf go1.22.4.linux-amd64.tar.gz
  rm go1.22.4.linux-amd64.tar.gz
fi
export PATH="/usr/local/go/bin:$HOME/go/bin:$PATH"
echo 'export PATH="/usr/local/go/bin:$HOME/go/bin:$PATH"' >> ~/.bashrc

echo "==> Go version: $(go version)"

echo "==> Clone infergo"
if [[ -d "$BUILD_DIR" ]]; then
  echo "    repo exists, pulling latest"
  git -C "$BUILD_DIR" fetch origin
  git -C "$BUILD_DIR" checkout "$BRANCH"
  git -C "$BUILD_DIR" pull origin "$BRANCH"
else
  git clone --depth 1 -b "$BRANCH" "$REPO_URL" "$BUILD_DIR"
fi

echo "==> Detect GPU count"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
echo "    GPUs detected: $GPU_COUNT"

echo "==> CMake configure (CUDA)"
cd "$BUILD_DIR"
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=OpenBLAS \
  -DCMAKE_CUDA_ARCHITECTURES=70  # V100=70, A100=80, T4=75
# Override if needed: -DCMAKE_CUDA_ARCHITECTURES=75 for T4, 80 for A100

echo "==> Build (j8)"
cmake --build build -j8

echo "==> Build Go binary"
go build -C go -o "$BUILD_DIR/infergo" ./cmd/infergo/

echo "==> Binary size: $(du -sh "$BUILD_DIR/infergo" | cut -f1)"

echo "==> Run ctests"
cd "$BUILD_DIR/build"
ctest --output-on-failure -j4 2>&1 | tail -5

echo "==> Model directory"
mkdir -p "$MODEL_DIR"
echo "    Place model at: $MODEL_DIR/llama3-8b-q4.gguf"
echo "    Download with:"
echo "      $BUILD_DIR/infergo pull bartowski/Meta-Llama-3-8B-Instruct-GGUF --quant Q4_K_M --out $MODEL_DIR/llama3-8b-q4.gguf"

echo ""
echo "==> Setup complete. Binary: $BUILD_DIR/infergo"
