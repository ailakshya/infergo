package llm_test

import (
	"testing"

	"github.com/ailakshya/infergo/llm"
)

func TestSpeculativeCreate(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 2048, 4, 512)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	// Use same model as both target and draft (same vocab guaranteed)
	sd, err := llm.NewSpeculativeDecoder(m, testModelPath, 99, 3)
	if err != nil {
		t.Fatalf("NewSpeculativeDecoder: %v", err)
	}
	defer sd.Close()
	t.Log("speculative decoder created OK")
}

func TestSpeculativeGenerate(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 2048, 4, 512)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	sd, err := llm.NewSpeculativeDecoder(m, testModelPath, 99, 3)
	if err != nil {
		t.Fatalf("NewSpeculativeDecoder: %v", err)
	}
	defer sd.Close()

	// Use a prompt that elicits output (chat template format for TinyLlama)
	prompt := "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is 2+2?</s>\n<|assistant|>\n"
	tokens, err := m.Tokenize(prompt, false, 256)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}

	text, stats, err := sd.Generate(tokens, 32, 0)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	t.Logf("output: %q", text)
	t.Logf("stats: predicted=%d, drafted=%d, accepted=%d, accept_rate=%.0f%%",
		stats.Predicted, stats.Drafted, stats.Accepted, stats.AcceptRate()*100)

	if stats.Predicted == 0 {
		t.Error("no tokens generated")
	}
}
