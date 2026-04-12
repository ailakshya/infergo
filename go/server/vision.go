package server

import (
	"context"
	"encoding/json"
	"net/http"
)

// VisionModel is a model that accepts image+text input (multimodal LLM).
type VisionModel interface {
	Model
	GenerateVision(ctx context.Context, prompt string, imageB64 string, maxTokens int, temp float32) (string, int, int, error)
}

// VisionChatRequest extends ChatCompletionRequest with image input.
// Compatible with OpenAI's vision API format.
type VisionChatMessage struct {
	Role    string              `json:"role"`
	Content json.RawMessage     `json:"content"` // string or array of content parts
}

// ContentPart is one part of a multimodal message (text or image_url).
type ContentPart struct {
	Type     string    `json:"type"`               // "text" or "image_url"
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL holds the URL or base64 data for an image.
type ImageURL struct {
	URL string `json:"url"` // "data:image/jpeg;base64,..." or HTTP URL
}

// handleVisionChat handles multimodal chat requests with image input.
// This is a stub — requires CLIP vision encoder integration.
func (s *Server) handleVisionChat(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented,
		"vision models require a multimodal model (e.g. LLaVA). "+
			"Load with: --model llava:model.gguf --vision-model mmproj.gguf")
}
