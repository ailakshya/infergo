# Security Guide

## Authentication

infergo supports API key authentication via Bearer tokens.

### Setup

```bash
# Set API key on server
infergo serve --model models/llama3-8b-q4.gguf --api-key YOUR_SECRET_KEY

# Or via environment variable
export INFERGO_API_KEY=YOUR_SECRET_KEY
infergo serve --model models/llama3-8b-q4.gguf
```

### Client usage

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Authorization: Bearer YOUR_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llm","messages":[{"role":"user","content":"Hello"}]}'
```

### Kubernetes Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: infergo-api-key
type: Opaque
stringData:
  api-key: "your-production-key-here"
---
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: infergo
          env:
            - name: INFERGO_API_KEY
              valueFrom:
                secretKeyRef:
                  name: infergo-api-key
                  key: api-key
```

---

## Rate Limiting

Prevent abuse with per-key and global rate limits.

```bash
infergo serve --model models/llama3-8b-q4.gguf \
  --api-key YOUR_KEY \
  --rate-limit 100        # 100 requests/minute global
  --rate-limit-per-key 20 # 20 requests/minute per API key
```

Rate-limited responses return HTTP 429 with `Retry-After` header.

---

## RBAC (Role-Based Access Control)

Three built-in roles control endpoint access:

| Role | Permissions |
|---|---|
| `admin` | All endpoints including `/v1/admin/*`, hot reload, tenant management |
| `user` | Inference endpoints: chat, embed, detect, search, RAG, NER, etc. |
| `readonly` | Read-only: `/v1/models`, `/health/*`, `/metrics` |

### Assigning roles

```bash
# In server config
infergo serve --model llm:model.gguf \
  --api-key admin:ADMIN_KEY \
  --api-key user:USER_KEY \
  --api-key readonly:READONLY_KEY
```

Unauthenticated requests (no API key set) default to `user` role.

---

## IP Allowlisting

Restrict access to specific IP addresses or CIDR ranges.

```bash
infergo serve --model models/llama3-8b-q4.gguf \
  --allow-ip 10.0.0.0/8 \
  --allow-ip 192.168.1.0/24 \
  --allow-ip 203.0.113.42
```

When enabled, requests from non-allowed IPs receive HTTP 403.

---

## Content Safety

### PII Detection and Redaction

infergo can detect and redact personally identifiable information in inputs and outputs:

- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- IP addresses

Enable via server config or per-request with `"redact_pii": true`.

### Content Filtering

Block harmful content categories:

- Violence
- Hate speech
- Self-harm
- Explicit content

---

## Audit Logging

All requests are logged in JSONL format with cryptographic hash chains for tamper evidence.

```bash
infergo serve --model models/llama3-8b-q4.gguf \
  --audit-log /var/log/infergo/audit.jsonl
```

Each log entry contains:
- Timestamp, request ID, client IP
- Model used, endpoint, response status
- SHA-256 hash linking to previous entry (tamper-evident chain)

Hash-only mode (no request/response bodies) available for sensitive environments:

```bash
--audit-log-mode hash-only
```

---

## GDPR Data Deletion

Delete all stored data for a specific user:

```bash
curl -X DELETE http://localhost:9090/v1/admin/gdpr/user123 \
  -H "Authorization: Bearer ADMIN_KEY"
```

This purges:
- Conversation memory
- Vector DB entries with matching user metadata
- Feedback records
- Audit log entries (marked as deleted)

---

## Model Encryption

Encrypt model files at rest with AES-256-GCM:

```bash
# Encrypt
infergo convert --input model.gguf --encrypt --key-file model.key

# Serve encrypted model
infergo serve --model models/model.gguf.enc --key-file model.key
```

---

## Network Security

### TLS

Use a reverse proxy (nginx, Caddy, Traefik) for TLS termination:

```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;

    location / {
        proxy_pass http://127.0.0.1:9090;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Unix Domain Socket (same machine)

For same-machine clients, use UDS to avoid network exposure entirely:

```bash
infergo serve --model models/llama3-8b-q4.gguf --uds /tmp/infergo.sock
```

### Shared Memory (lowest latency)

For maximum security and performance on same machine:

```bash
infergo serve --model models/llama3-8b-q4.gguf --shm infergo_shm
```

No network interface exposed. Communication via kernel-managed shared memory.

---

## Webhook Security

Webhooks are signed with HMAC-SHA256:

```bash
infergo serve --model models/llama3-8b-q4.gguf \
  --webhook-url https://your-service.com/hook \
  --webhook-secret YOUR_WEBHOOK_SECRET
```

Verify webhook signatures in your handler:

```python
import hmac, hashlib

def verify(body, signature, secret):
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)
```

---

## Security Checklist

| Item | Action |
|---|---|
| Set API key | `--api-key` or `INFERGO_API_KEY` env var |
| Enable TLS | Reverse proxy with SSL certificate |
| Restrict IPs | `--allow-ip` for internal networks |
| Set rate limits | `--rate-limit` and `--rate-limit-per-key` |
| Enable audit log | `--audit-log /path/to/audit.jsonl` |
| Use RBAC | Assign `admin:`/`user:`/`readonly:` key prefixes |
| Redact PII | Enable PII detection in config |
| Secure models | `--encrypt` for model files at rest |
| Use UDS/SHM | Avoid TCP when client is on same machine |
