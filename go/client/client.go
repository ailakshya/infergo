package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client is a typed infergo API client.
type Client struct {
	baseURL    string
	httpClient *http.Client
	apiKey     string
}

// Option configures a Client.
type Option func(*Client)

// WithAPIKey sets the Authorization: Bearer header on all requests.
func WithAPIKey(key string) Option {
	return func(c *Client) {
		c.apiKey = key
	}
}

// WithHTTPClient overrides the default HTTP client.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		c.httpClient = hc
	}
}

// WithTimeout sets a per-request timeout (default: 120s).
func WithTimeout(d time.Duration) Option {
	return func(c *Client) {
		c.httpClient = &http.Client{Timeout: d}
	}
}

// New creates a new Client pointing at baseURL (e.g. "http://localhost:9090").
func New(baseURL string, opts ...Option) *Client {
	c := &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// ─── Chat ────────────────────────────────────────────────────────────────────

// Message is a single turn in a conversation.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest holds the parameters for a chat completion call.
type ChatRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	MaxTokens int       `json:"max_tokens,omitempty"`
	Temp      float32   `json:"temperature,omitempty"`
}

// ChatResponse holds the result of a blocking chat completion.
type ChatResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Content string // extracted from choices[0].message.content
	Usage   struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	}
}

// chatCompletionRequest is the wire format sent to the server.
type chatCompletionRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	MaxTokens int       `json:"max_tokens,omitempty"`
	Temp      float32   `json:"temperature,omitempty"`
	Stream    bool      `json:"stream"`
}

// chatCompletionResponse is the wire format received from the server.
type chatCompletionResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// Chat sends a blocking chat completion request and returns the full response.
func (c *Client) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	wireReq := chatCompletionRequest{
		Model:     req.Model,
		Messages:  req.Messages,
		MaxTokens: req.MaxTokens,
		Temp:      req.Temp,
		Stream:    false,
	}
	var wireResp chatCompletionResponse
	if err := c.doJSON(ctx, http.MethodPost, "/v1/chat/completions", wireReq, &wireResp); err != nil {
		return nil, err
	}
	resp := &ChatResponse{
		ID:    wireResp.ID,
		Model: wireResp.Model,
	}
	resp.Usage.PromptTokens = wireResp.Usage.PromptTokens
	resp.Usage.CompletionTokens = wireResp.Usage.CompletionTokens
	if len(wireResp.Choices) > 0 {
		resp.Content = wireResp.Choices[0].Message.Content
	}
	return resp, nil
}

// ChatStream sends a streaming chat request and returns a channel of tokens
// and a channel of errors. Both channels are closed when the stream ends or
// ctx is cancelled. Reads SSE data: lines and parses delta.content.
func (c *Client) ChatStream(ctx context.Context, req ChatRequest) (<-chan string, <-chan error) {
	tokens := make(chan string, 32)
	errc := make(chan error, 1)

	wireReq := chatCompletionRequest{
		Model:     req.Model,
		Messages:  req.Messages,
		MaxTokens: req.MaxTokens,
		Temp:      req.Temp,
		Stream:    true,
	}

	go func() {
		defer close(tokens)
		defer close(errc)

		body, err := json.Marshal(wireReq)
		if err != nil {
			errc <- fmt.Errorf("marshal request: %w", err)
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			errc <- fmt.Errorf("create request: %w", err)
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")
		if c.apiKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
		}

		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			errc <- fmt.Errorf("http request: %w", err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(resp.Body)
			errc <- fmt.Errorf("server returned %d: %s", resp.StatusCode, string(b))
			return
		}

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "[DONE]" {
				break
			}
			var chunk struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
					FinishReason *string `json:"finish_reason"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				continue
			}
			if len(chunk.Choices) > 0 {
				if tok := chunk.Choices[0].Delta.Content; tok != "" {
					select {
					case tokens <- tok:
					case <-ctx.Done():
						return
					}
				}
			}
		}
		if err := scanner.Err(); err != nil && ctx.Err() == nil {
			errc <- fmt.Errorf("reading stream: %w", err)
		}
	}()

	return tokens, errc
}

// ─── Embeddings ──────────────────────────────────────────────────────────────

// EmbedRequest holds the parameters for an embedding call.
type EmbedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// embeddingResponse is the wire format received from the server.
type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Embed returns a float32 embedding vector for the given input.
func (c *Client) Embed(ctx context.Context, req EmbedRequest) ([]float32, error) {
	var resp embeddingResponse
	if err := c.doJSON(ctx, http.MethodPost, "/v1/embeddings", req, &resp); err != nil {
		return nil, err
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data in response")
	}
	return resp.Data[0].Embedding, nil
}

// ─── Detection ───────────────────────────────────────────────────────────────

// DetectRequest holds the parameters for an object detection call.
type DetectRequest struct {
	Model      string  `json:"model"`
	ImageB64   string  `json:"image_b64"`
	ConfThresh float32 `json:"conf_thresh,omitempty"`
	IouThresh  float32 `json:"iou_thresh,omitempty"`
}

// Detection is a single bounding box result.
type Detection struct {
	X1, Y1, X2, Y2 float32
	ClassID         int
	Confidence      float32
}

// DetectResponse holds the result of a detection call.
type DetectResponse struct {
	Model   string      `json:"model"`
	Objects []Detection `json:"objects"`
}

// Detect sends a detection request and returns bounding boxes.
func (c *Client) Detect(ctx context.Context, req DetectRequest) (*DetectResponse, error) {
	var resp DetectResponse
	if err := c.doJSON(ctx, http.MethodPost, "/v1/detect", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ─── Models ──────────────────────────────────────────────────────────────────

// ModelInfo holds metadata about a loaded model.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
}

// modelListResponse is the wire format received from GET /v1/models.
type modelListResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// ListModels returns all loaded models from GET /v1/models.
func (c *Client) ListModels(ctx context.Context) ([]ModelInfo, error) {
	var resp modelListResponse
	if err := c.doJSON(ctx, http.MethodGet, "/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.Data, nil
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

// doJSON marshals body (if non-nil) as JSON, performs the HTTP request, checks
// the status code, and unmarshals the response into out.
func (c *Client) doJSON(ctx context.Context, method, path string, body any, out any) error {
	var bodyReader io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, bodyReader)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Accept", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("http %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("server returned %d: %s", resp.StatusCode, string(respBody))
	}

	if out != nil {
		if err := json.Unmarshal(respBody, out); err != nil {
			return fmt.Errorf("unmarshal response: %w", err)
		}
	}
	return nil
}
