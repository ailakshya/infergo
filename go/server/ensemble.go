package server

import (
	"context"
	"sync"
)

// EnsembleResult holds responses from multiple models. OPT-122.
type EnsembleResult struct {
	Responses []struct {
		Model   string `json:"model"`
		Content string `json:"content"`
	} `json:"responses"`
	Selected string `json:"selected"` // which response was chosen
	Method   string `json:"method"`   // "longest", "first", "vote"
}

// RunEnsemble queries N models in parallel and picks the best response.
func RunEnsemble(ctx context.Context, models []LLMModel, modelNames []string, prompt string, maxTokens int, temp float32) EnsembleResult {
	type result struct {
		idx     int
		content string
		err     error
	}

	results := make([]result, len(models))
	var wg sync.WaitGroup

	for i, model := range models {
		wg.Add(1)
		go func(idx int, m LLMModel) {
			defer wg.Done()
			text, _, _, err := m.Generate(ctx, prompt, maxTokens, temp)
			results[idx] = result{idx: idx, content: text, err: err}
		}(i, model)
	}
	wg.Wait()

	// Pick best response (simple: longest non-error response)
	var ensemble EnsembleResult
	ensemble.Method = "longest"
	bestIdx := 0
	bestLen := 0

	for i, r := range results {
		if r.err != nil {
			continue
		}
		ensemble.Responses = append(ensemble.Responses, struct {
			Model   string `json:"model"`
			Content string `json:"content"`
		}{Model: modelNames[i], Content: r.content})

		if len(r.content) > bestLen {
			bestLen = len(r.content)
			bestIdx = i
		}
	}

	if bestIdx < len(modelNames) {
		ensemble.Selected = modelNames[bestIdx]
	}

	return ensemble
}
