package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// ── OPT-76: Named Entity Recognition ────────────────────────────────────────

type NERRequest struct {
	Model string `json:"model"`
	Text  string `json:"text"`
}

type NEREntity struct {
	Text string `json:"text"`
	Type string `json:"type"` // PERSON, ORG, LOCATION, DATE, etc.
}

type NERResponse struct {
	Entities []NEREntity `json:"entities"`
}

func (s *Server) handleNER(w http.ResponseWriter, r *http.Request) {
	var req NERRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	llm, ok := ref.Model.(LLMModel)
	if !ok {
		writeError(w, http.StatusBadRequest, "model does not support NER")
		return
	}

	prompt := fmt.Sprintf(`<|im_start|>system
Extract named entities from the text. Return JSON array of objects with "text" and "type" fields.
Types: PERSON, ORG, LOCATION, DATE, MONEY, PRODUCT.
Return ONLY the JSON array, no other text.<|im_end|>
<|im_start|>user
%s<|im_end|>
<|im_start|>assistant
`, req.Text)

	ctx := WithGrammar(r.Context(), JSONGrammar)
	text, _, _, err := llm.Generate(ctx, prompt, 256, 0.1)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	var entities []NEREntity
	json.Unmarshal([]byte(text), &entities)
	writeJSON(w, http.StatusOK, NERResponse{Entities: entities})
}

// ── OPT-77: Sentiment Analysis ──────────────────────────────────────────────

type SentimentRequest struct {
	Model string   `json:"model"`
	Text  string   `json:"text"`
	Texts []string `json:"texts,omitempty"` // batch mode
}

type SentimentResult struct {
	Sentiment string  `json:"sentiment"` // positive, negative, neutral
	Score     float64 `json:"score"`
}

type SentimentResponse struct {
	Results []SentimentResult `json:"results"`
}

func (s *Server) handleSentiment(w http.ResponseWriter, r *http.Request) {
	var req SentimentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	texts := req.Texts
	if req.Text != "" {
		texts = append([]string{req.Text}, texts...)
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	llm, ok := ref.Model.(LLMModel)
	if !ok {
		writeError(w, http.StatusBadRequest, "model does not support sentiment")
		return
	}

	var results []SentimentResult
	for _, text := range texts {
		prompt := fmt.Sprintf(`<|im_start|>system
Analyze sentiment. Return JSON: {"sentiment": "positive"|"negative"|"neutral", "score": 0.0-1.0}
Return ONLY the JSON object.<|im_end|>
<|im_start|>user
%s<|im_end|>
<|im_start|>assistant
`, text)

		ctx := WithGrammar(r.Context(), JSONGrammar)
		out, _, _, err := llm.Generate(ctx, prompt, 64, 0.1)
		if err != nil {
			results = append(results, SentimentResult{Sentiment: "unknown", Score: 0})
			continue
		}

		var result SentimentResult
		if json.Unmarshal([]byte(out), &result) != nil {
			result = SentimentResult{Sentiment: "unknown", Score: 0}
		}
		results = append(results, result)
	}

	writeJSON(w, http.StatusOK, SentimentResponse{Results: results})
}

// ── OPT-78: Text Classification ─────────────────────────────────────────────

type ClassifyRequest struct {
	Model      string   `json:"model"`
	Text       string   `json:"text"`
	Labels     []string `json:"labels"`
	MultiLabel bool     `json:"multi_label,omitempty"`
}

type ClassifyResult struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
}

type ClassifyResponse struct {
	Results []ClassifyResult `json:"results"`
}

func (s *Server) handleClassify(w http.ResponseWriter, r *http.Request) {
	var req ClassifyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Labels) == 0 {
		writeError(w, http.StatusBadRequest, "labels required")
		return
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	llm, ok := ref.Model.(LLMModel)
	if !ok {
		writeError(w, http.StatusBadRequest, "model does not support classification")
		return
	}

	labelsStr := ""
	for i, l := range req.Labels {
		if i > 0 {
			labelsStr += ", "
		}
		labelsStr += `"` + l + `"`
	}

	prompt := fmt.Sprintf(`<|im_start|>system
Classify the text into one of these labels: [%s].
Return JSON: {"label": "<label>", "confidence": 0.0-1.0}
Return ONLY the JSON object.<|im_end|>
<|im_start|>user
%s<|im_end|>
<|im_start|>assistant
`, labelsStr, req.Text)

	ctx := WithGrammar(r.Context(), JSONGrammar)
	out, _, _, err := llm.Generate(ctx, prompt, 64, 0.1)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	var result ClassifyResult
	json.Unmarshal([]byte(out), &result)
	writeJSON(w, http.StatusOK, ClassifyResponse{Results: []ClassifyResult{result}})
}

// ── OPT-80: Summarization ───────────────────────────────────────────────────

type SummarizeRequest struct {
	Model     string `json:"model"`
	Text      string `json:"text"`
	MaxLength int    `json:"max_length,omitempty"` // max words in summary
	Mode      string `json:"mode,omitempty"`       // "abstractive" (default) or "extractive"
}

type SummarizeResponse struct {
	Summary string `json:"summary"`
	Mode    string `json:"mode"`
}

func (s *Server) handleSummarize(w http.ResponseWriter, r *http.Request) {
	var req SummarizeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	llm, ok := ref.Model.(LLMModel)
	if !ok {
		writeError(w, http.StatusBadRequest, "model does not support summarization")
		return
	}

	maxLen := req.MaxLength
	if maxLen <= 0 {
		maxLen = 100
	}
	mode := req.Mode
	if mode == "" {
		mode = "abstractive"
	}

	var prompt string
	if mode == "extractive" {
		prompt = fmt.Sprintf(`<|im_start|>system
Extract the most important sentences from the text. Return at most %d words.
Return ONLY the extracted sentences, no other text.<|im_end|>
<|im_start|>user
%s<|im_end|>
<|im_start|>assistant
`, maxLen, req.Text)
	} else {
		prompt = fmt.Sprintf(`<|im_start|>system
Summarize the text concisely in at most %d words.
Return ONLY the summary, no other text.<|im_end|>
<|im_start|>user
%s<|im_end|>
<|im_start|>assistant
`, maxLen, req.Text)
	}

	ctx := context.Background()
	maxToks := maxLen * 2 // rough word-to-token ratio
	if maxToks > 512 {
		maxToks = 512
	}
	text, _, _, err := llm.Generate(ctx, prompt, maxToks, 0.3)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, SummarizeResponse{Summary: text, Mode: mode})
}
