package server

import (
	"context"
	"net/http"
)

// DiffusionModel generates images from text prompts.
type DiffusionModel interface {
	Model
	Generate(ctx context.Context, prompt string, negPrompt string, width, height, steps int, seed int64) ([]byte, error)
}

// ImageGenerationRequest mirrors the OpenAI /v1/images/generations body.
type ImageGenerationRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n,omitempty"`
	Size           string `json:"size,omitempty"` // "512x512", "1024x1024"
	Quality        string `json:"quality,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"` // "url" or "b64_json"
}

// ImageGenerationResponse mirrors the OpenAI response.
type ImageGenerationResponse struct {
	Created int64       `json:"created"`
	Data    []ImageData `json:"data"`
}

// ImageData is one generated image.
type ImageData struct {
	URL     string `json:"url,omitempty"`
	B64JSON string `json:"b64_json,omitempty"`
}

// handleImageGeneration handles POST /v1/images/generations.
func (s *Server) handleImageGeneration(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented,
		"image generation requires a diffusion model (e.g. Stable Diffusion). "+
			"Load with: --model sd:stable-diffusion-v1-5.safetensors")
}
