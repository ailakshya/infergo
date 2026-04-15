package server

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"time"
)

// WebhookSender sends HTTP POST callbacks on batch completion.
type WebhookSender struct {
	client    *http.Client
	secret    string // HMAC secret for signing
	maxRetry  int
}

// NewWebhookSender creates a webhook sender.
func NewWebhookSender(secret string) *WebhookSender {
	return &WebhookSender{
		client:   &http.Client{Timeout: 10 * time.Second},
		secret:   secret,
		maxRetry: 3,
	}
}

// Send posts a JSON payload to the webhook URL with HMAC signature.
func (w *WebhookSender) Send(url string, payload interface{}) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	var lastErr error
	for attempt := 0; attempt <= w.maxRetry; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(attempt*attempt) * time.Second) // exponential backoff
		}

		req, err := http.NewRequest("POST", url, bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")

		// HMAC signature
		if w.secret != "" {
			mac := hmac.New(sha256.New, []byte(w.secret))
			mac.Write(body)
			sig := hex.EncodeToString(mac.Sum(nil))
			req.Header.Set("X-Webhook-Signature", sig)
		}

		resp, err := w.client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		resp.Body.Close()

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return nil
		}
		lastErr = &webhookError{StatusCode: resp.StatusCode}
	}
	return lastErr
}

type webhookError struct {
	StatusCode int
}

func (e *webhookError) Error() string {
	return "webhook failed with status " + http.StatusText(e.StatusCode)
}
