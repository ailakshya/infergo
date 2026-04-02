# Deployment Guide

---

## Docker

### CPU image

```bash
docker build -f Dockerfile.cpu -t infergo:cpu .

docker run --rm -p 9090:9090 \
  -v /path/to/models:/models:ro \
  infergo:cpu \
  serve --model /models/llama3-8b-q4.gguf --port 9090
```

### CUDA image

Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker build -f Dockerfile.cuda -t infergo:cuda .

docker run --rm --gpus all -p 9090:9090 \
  -v /path/to/models:/models:ro \
  infergo:cuda \
  serve --model /models/llama3-8b-q4.gguf --provider cuda --gpu-layers 999 --port 9090
```

### docker-compose

```bash
# CPU
docker compose up infergo-cpu

# CUDA
docker compose up infergo-cuda
```

Models go in `./models/` relative to the repo root (mounted read-only).

---

## Kubernetes

### Deployment manifest (CPU)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: infergo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: infergo
  template:
    metadata:
      labels:
        app: infergo
    spec:
      containers:
        - name: infergo
          image: infergo:cpu
          args:
            - serve
            - --model=/models/llama3-8b-q4.gguf
            - --port=9090
          ports:
            - containerPort: 9090
          volumeMounts:
            - name: models
              mountPath: /models
              readOnly: true
          livenessProbe:
            httpGet:
              path: /healthz
              port: 9090
            initialDelaySeconds: 10
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /readyz
              port: 9090
            initialDelaySeconds: 5
            periodSeconds: 10
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: infergo-models
---
apiVersion: v1
kind: Service
metadata:
  name: infergo
spec:
  selector:
    app: infergo
  ports:
    - port: 80
      targetPort: 9090
```

### GPU node (CUDA)

Add resource requests and node selector:

```yaml
      nodeSelector:
        accelerator: nvidia-tesla-a100     # or whatever your node label is
      containers:
        - name: infergo
          image: infergo:cuda
          args:
            - serve
            - --model=/models/llama3-8b-q4.gguf
            - --provider=cuda
            - --gpu-layers=999
            - --port=9090
          resources:
            limits:
              nvidia.com/gpu: "1"
```

### Horizontal scaling note

Each infergo pod loads its own copy of the model into GPU memory. For very large models (>10B params) prefer vertical scaling (fewer pods, more GPU memory) over horizontal.

---

## Bare metal

### Systemd service

```ini
# /etc/systemd/system/infergo.service
[Unit]
Description=infergo inference server
After=network.target

[Service]
Type=simple
User=infergo
ExecStart=/usr/local/bin/infergo serve \
    --model /opt/models/llama3-8b-q4.gguf \
    --provider cuda \
    --gpu-layers 999 \
    --port 9090
Restart=on-failure
RestartSec=5
Environment=LD_LIBRARY_PATH=/usr/local/lib/infergo

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now infergo
journalctl -u infergo -f
```

### Nginx reverse proxy (optional)

```nginx
upstream infergo {
    server 127.0.0.1:9090;
}

server {
    listen 443 ssl;
    server_name infergo.example.com;

    location / {
        proxy_pass http://infergo;
        proxy_set_header Host $host;
        # Required for SSE streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 120s;
    }
}
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LD_LIBRARY_PATH` | — | Must include the directory containing `libinfer_api.so` |
| `CUDA_VISIBLE_DEVICES` | all | Restrict to specific GPU indices |

---

## Prometheus + Grafana

infergo exposes Prometheus metrics at `/metrics`. Add a scrape config:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: infergo
    static_configs:
      - targets: ["infergo:9090"]
```

Key metrics:

| Metric | Type | Description |
|---|---|---|
| `infergo_requests_total` | Counter | Requests by model, endpoint, status |
| `infergo_request_duration_seconds` | Histogram | Latency by model and endpoint |
| `infergo_batch_size` | Histogram | Batch sizes dispatched |
| `infergo_tokens_per_second` | Gauge | Generation throughput per model |
| `infergo_gpu_memory_bytes` | Gauge | GPU VRAM usage per device |
