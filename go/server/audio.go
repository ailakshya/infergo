package server

import (
	"context"
	"net/http"
)

// AudioModel is a model that transcribes audio to text (Whisper-compatible).
type AudioModel interface {
	Model
	Transcribe(ctx context.Context, audioBytes []byte, language string) (string, error)
}

// TranscriptionResponse mirrors the OpenAI /v1/audio/transcriptions response.
type TranscriptionResponse struct {
	Text string `json:"text"`
}

// handleTranscription handles POST /v1/audio/transcriptions.
// Accepts multipart/form-data with an audio file.
// This is a stub — requires Whisper GGML model integration.
func (s *Server) handleTranscription(w http.ResponseWriter, r *http.Request) {
	writeError(w, http.StatusNotImplemented,
		"audio transcription requires a Whisper model. "+
			"Load with: --model whisper:ggml-base.en.bin")
}
