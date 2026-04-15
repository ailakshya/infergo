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
        accelerator: nvidia-tesla-a100
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

## Multi-tenant configuration

infergo supports per-API-key resource isolation through the tenant system.

### Setting up tenants

Create tenants via the admin API:

```bash
curl -X POST http://localhost:9090/v1/admin/tenants \
  -H "Authorization: Bearer admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "tenant-acme",
    "api_key": "acme-api-key-123",
    "allowed_models": ["llama3-8b-q4", "embed"],
    "rate_limit": 10.0,
    "max_tokens": 4096,
    "monthly_quota": 1000000
  }'
```

### Tenant configuration fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique tenant identifier |
| `api_key` | string | API key for this tenant |
| `allowed_models` | []string | Models this tenant can access (empty = all) |
| `rate_limit` | float | Max requests per second (0 = unlimited) |
| `max_tokens` | int | Max tokens per request (0 = unlimited) |
| `monthly_quota` | int | Total tokens per month (0 = unlimited) |

### Monitoring tenant usage

```bash
curl http://localhost:9090/v1/admin/tenants/tenant-acme/usage \
  -H "Authorization: Bearer admin-key"
```

Returns:

```json
{
  "tenant_id": "tenant-acme",
  "total_tokens": 45230,
  "request_count": 127,
  "month_start": "2026-04-01T00:00:00Z",
  "quota_used_pct": 4.52
}
```

### Go setup

```go
tenants := server.NewTenantStore()
tenants.Upsert(server.TenantConfig{
    ID:            "tenant-acme",
    APIKey:        "acme-api-key-123",
    AllowedModels: []string{"llama3-8b-q4"},
    RateLimit:     10.0,
    MonthlyQuota:  1_000_000,
})
srv.SetTenants(tenants)
```

---

## RBAC (Role-Based Access Control)

### Three roles

| Role | Access |
|---|---|
| `admin` | All endpoints including `/v1/admin/*` |
| `user` | Inference endpoints only |
| `readonly` | Read-only: models list, health, metrics, UI |

### Configuration

Pass API key to role mappings via environment or config:

```bash
./infergo serve \
  --model models/llama3-8b-q4.gguf \
  --api-key admin-key-123 \
  --port 9090
```

### Go setup

```go
rbac := server.NewRBACConfig(map[string]string{
    "admin-key-123":   "admin",
    "team-key-456":    "user",
    "monitor-key-789": "readonly",
})
handler := server.RBACMiddleware(rbac)(srv)
http.ListenAndServe(":9090", handler)
```

### Making requests

```bash
# Admin can access everything
curl http://localhost:9090/v1/admin/tenants \
  -H "Authorization: Bearer admin-key-123"

# User can run inference but not admin endpoints
curl http://localhost:9090/v1/chat/completions \
  -H "Authorization: Bearer team-key-456" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b-q4","messages":[{"role":"user","content":"Hello"}]}'

# Readonly can list models and check health
curl http://localhost:9090/v1/models \
  -H "Authorization: Bearer monitor-key-789"
```

---

## Canary deployments

Deploy a new model version and gradually shift traffic while monitoring error rates.

### Create a canary deployment

```bash
curl -X POST http://localhost:9090/v1/admin/canary \
  -H "Authorization: Bearer admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "llama3-8b-q4",
    "new_model": "llama3-8b-q4-v2",
    "traffic_pct": 0.10,
    "max_error_rate": 0.05,
    "auto_promote_after": 1000
  }'
```

### How it works

1. `traffic_pct` of requests to `base_model` are routed to `new_model`
2. If the canary error rate exceeds `max_error_rate`, automatic rollback triggers
3. After `auto_promote_after` successful canary requests, the new model is promoted
4. All traffic then goes to the new model

### Monitor canary status

```bash
curl http://localhost:9090/v1/admin/canary \
  -H "Authorization: Bearer admin-key"
```

### Cancel a canary

```bash
curl -X DELETE http://localhost:9090/v1/admin/canary \
  -H "Authorization: Bearer admin-key"
```

---

## A/B testing

Compare two models by splitting traffic:

```go
ab := server.NewABTest("model-a", "model-b", 0.5) // 50% split
a_count, b_count := ab.Stats()
ab.Disable() // stop the test
```

The A/B test routes requests for `model-a` to either model based on the split percentage and tracks traffic counts for analysis.

---

## Circuit breaker configuration

Automatically disable a failing model and recover after a cooldown period.

### Go setup

```go
cb := server.NewCircuitBreaker(5, 30)
// threshold=5: open circuit after 5 consecutive failures
// cooldown=30: wait 30 seconds before trying half-open
```

### State transitions

```
closed (normal) --[5 failures]--> open (disabled) --[30s cooldown]--> half-open (testing)
     ^                                                                      |
     |-------------------[success]------------------------------------------|
     |-------------------[failure]-->  open (disabled) ----------------------|
```

When the circuit is open, all requests to that model return 503 immediately. After the cooldown, one test request is allowed through. If it succeeds, the circuit closes. If it fails, the circuit reopens.

---

## PII mode configuration

Detect and handle personal identifiable information in requests.

### Via API

```bash
# Get current PII config
curl http://localhost:9090/v1/admin/pii \
  -H "Authorization: Bearer admin-key"

# Set PII mode
curl -X POST http://localhost:9090/v1/admin/pii \
  -H "Authorization: Bearer admin-key" \
  -H "Content-Type: application/json" \
  -d '{"mode": "redact"}'
```

### Modes

| Mode | Behavior |
|---|---|
| `block` | Reject the request with HTTP 400 |
| `redact` | Replace PII with `[REDACTED]` before sending to the model |
| `log` | Log a warning but process the request normally |

### Detected PII types

| Type | Pattern |
|---|---|
| `EMAIL` | Email addresses |
| `PHONE` | US phone numbers |
| `SSN` | Social Security Numbers |
| `CREDIT_CARD` | Credit card numbers |
| `IP_ADDRESS` | IPv4 addresses |

### Go setup

```go
pii := server.NewPIIDetector("redact")
srv.SetPII(pii)
```

---

## Audit logging setup

Write structured JSONL audit logs for compliance and debugging.

### Go setup

```go
logger, err := server.NewAuditLogger("/var/log/infergo/audit.jsonl", false)
// hashOnly=false: store full prompt text
// hashOnly=true: store only prompt hash (for privacy)
```

### Log entry format

Each line in the JSONL file contains:

```json
{
  "timestamp": "2026-04-13T10:30:00Z",
  "api_key": "...k123",
  "model": "llama3-8b-q4",
  "prompt_hash": "a1b2c3d4e5f67890",
  "prompt": "What is the meaning of life?",
  "tokens": 42,
  "latency_ms": 234,
  "status": 200
}
```

When `hashOnly` is true, the `prompt` field is omitted and only the hash is stored.

---

## Data retention policy

Automatically clean up old data files (audit logs, cached responses, session data).

### Go setup

```go
retention := server.NewRetentionPolicy(30, "/var/log/infergo/", "/var/lib/infergo/cache/")
// 30 days retention, cleaning two directories

stopCh := make(chan struct{})
retention.StartPeriodicCleanup(24 * time.Hour, stopCh)
// Runs cleanup daily

// On shutdown:
close(stopCh)
```

Files older than the retention period are automatically deleted during each cleanup cycle.

---

## Content guardrails

Configure content safety filters to block or modify unsafe content.

```bash
# Get current guardrail config
curl http://localhost:9090/v1/admin/guardrails \
  -H "Authorization: Bearer admin-key"

# Update guardrails
curl -X POST http://localhost:9090/v1/admin/guardrails \
  -H "Authorization: Bearer admin-key" \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "block_categories": ["violence", "hate"]}'
```

---

## GDPR compliance

Delete all data associated with a user:

```bash
curl -X DELETE http://localhost:9090/v1/admin/gdpr/user-id-123 \
  -H "Authorization: Bearer admin-key"
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LD_LIBRARY_PATH` | -- | Must include the directory containing `libinfer_api.so` |
| `CUDA_VISIBLE_DEVICES` | all | Restrict to specific GPU indices |
| `INFERGO_API_KEY` | (none) | API key (alternative to `--api-key` flag) |

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

---

## Distributed tracing (OpenTelemetry)

Enable distributed tracing with the OTLP exporter:

```bash
./infergo serve \
  --model models/llama3-8b-q4.gguf \
  --otlp-endpoint localhost:4318 \
  --port 9090
```

Traces are exported to any OTLP-compatible backend (Jaeger, Zipkin, Grafana Tempo).

---

## Prefill-decode separation

For large-scale deployments, separate prefill (prompt processing) from decode (token generation) across different machines:

```bash
# Prefill node (processes prompts, serializes KV cache)
./infergo serve --model models/llama3-8b-q4.gguf --mode prefill --port 9090

# Decode node (receives KV cache, generates tokens)
./infergo serve --model models/llama3-8b-q4.gguf --mode decode --port 9091
```

In prefill mode, only `/v1/prefill` is active. In decode mode, only `/v1/decode` is active. In combined mode (default), all endpoints are active.

---

## Multi-GPU

### Tensor parallelism

Split model weights across multiple GPUs:

```bash
# Auto-detect GPUs and split evenly
./infergo serve --model models/llama3-70b-q4.gguf --tensor-split auto

# Custom split ratio (60% GPU 0, 40% GPU 1)
./infergo serve --model models/llama3-70b-q4.gguf --tensor-split 0.6,0.4
```

### Pipeline parallelism

Distribute layers across GPUs:

```bash
# 2-GPU pipeline
./infergo serve --model models/llama3-70b-q4.gguf --pipeline-stages 2
```

---

## Speculative decoding

Use a small draft model to speed up generation:

```bash
./infergo serve \
  --model models/llama3-70b-q4.gguf \
  --draft-model models/llama3-1b-q4.gguf \
  --n-draft 5 \
  --provider cuda --gpu-layers 999
```

The draft model generates candidate tokens that the target model verifies in a single forward pass, accepting correct predictions and rejecting incorrect ones.

---

## gRPC

infergo supports gRPC in addition to HTTP:

```bash
./infergo serve --model models/llama3-8b-q4.gguf --port 9090 --grpc-port 9091
```

Set `--grpc-port 0` to disable gRPC.
