package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
)

// PromptOptRequest is the body for POST /v1/admin/optimize-prompt. OPT-70.
type PromptOptRequest struct {
	Model       string `json:"model"`
	Task        string `json:"task"`        // description of the task
	TestInput   string `json:"test_input"`  // example input to test with
	Criteria    string `json:"criteria"`    // evaluation criteria
	NumVariants int    `json:"num_variants,omitempty"` // number of variants to generate
}

type PromptVariant struct {
	Prompt string  `json:"prompt"`
	Score  float64 `json:"score"`
	Output string  `json:"output"`
}

type PromptOptResponse struct {
	Best     PromptVariant   `json:"best"`
	Variants []PromptVariant `json:"variants"`
}

func (s *Server) handlePromptOptimize(w http.ResponseWriter, r *http.Request) {
	var req PromptOptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.NumVariants <= 0 {
		req.NumVariants = 5
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	llm, ok := ref.Model.(LLMModel)
	if !ok {
		writeError(w, http.StatusBadRequest, "model does not support generation")
		return
	}

	// Step 1: Generate prompt variants
	genPrompt := fmt.Sprintf(`<|im_start|>system
Generate %d different system prompts for the following task. Each should be a different approach.
Return JSON array: [{"prompt": "..."}, ...]
Task: %s
Criteria: %s<|im_end|>
<|im_start|>assistant
`, req.NumVariants, req.Task, req.Criteria)

	ctx := WithGrammar(context.Background(), JSONGrammar)
	variantsText, _, _, err := llm.Generate(ctx, genPrompt, 512, 0.8)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to generate variants: "+err.Error())
		return
	}

	var rawVariants []struct{ Prompt string }
	json.Unmarshal([]byte(variantsText), &rawVariants)

	// Step 2: Test each variant
	var variants []PromptVariant
	for _, rv := range rawVariants {
		testPrompt := fmt.Sprintf(`<|im_start|>system
%s<|im_end|>
<|im_start|>user
%s<|im_end|>
<|im_start|>assistant
`, rv.Prompt, req.TestInput)

		output, _, _, err := llm.Generate(context.Background(), testPrompt, 128, 0.3)
		if err != nil {
			continue
		}

		// Score: length as proxy (longer = more detailed, simple heuristic)
		score := float64(len(output)) / 500.0
		if score > 1.0 {
			score = 1.0
		}

		variants = append(variants, PromptVariant{
			Prompt: rv.Prompt,
			Score:  score,
			Output: output,
		})
	}

	sort.Slice(variants, func(i, j int) bool {
		return variants[i].Score > variants[j].Score
	})

	resp := PromptOptResponse{Variants: variants}
	if len(variants) > 0 {
		resp.Best = variants[0]
	}

	writeJSON(w, http.StatusOK, resp)
}
