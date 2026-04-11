package llm_test

import (
	"encoding/json"
	"testing"

	"github.com/ailakshya/infergo/llm"
	"github.com/ailakshya/infergo/server"
)

// ─── Sampler creation ────────────────────────────────────────────────────────

func TestNewGrammarSamplerNilModel(t *testing.T) {
	_, err := llm.NewGrammarSampler(nil, "root ::= \"test\"", "", 0.8, 0.9, 0, 0)
	if err == nil {
		t.Fatal("expected error for nil model")
	}
}

func TestNewGrammarSamplerEmptyGrammar(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 2, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	_, err = llm.NewGrammarSampler(m, "", "", 0.8, 0.9, 0, 0)
	if err == nil {
		t.Fatal("expected error for empty grammar")
	}
}

func TestNewGrammarSamplerCreate(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 2, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	smpl, err := llm.NewGrammarSampler(m, server.JSONGrammar, "root", 0.8, 0.9, 0, 0)
	if err != nil {
		t.Fatalf("NewGrammarSampler: %v", err)
	}
	smpl.Close()
	smpl.Close() // double close must not panic
}

// ─── Grammar-constrained generation ─────────────────────────────────────────

func TestGrammarSamplerJSONOutput(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 1024, 2, 512)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	// Tokenize a prompt asking for JSON output.
	tokens, err := m.Tokenize("Output a JSON object with key 'name' and value 'test':", true, 256)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}

	seq, err := m.NewSequence(tokens)
	if err != nil {
		t.Fatalf("NewSequence: %v", err)
	}
	defer seq.Close()

	smpl, err := llm.NewGrammarSampler(m, server.JSONGrammar, "root", 0.8, 0.9, 0, 42)
	if err != nil {
		t.Fatalf("NewGrammarSampler: %v", err)
	}
	defer smpl.Close()

	// Generate up to 64 tokens using zero-copy grammar-constrained sampling.
	var output []byte
	for i := 0; i < 64; i++ {
		if err := m.BatchDecode([]*llm.Sequence{seq}); err != nil {
			t.Fatalf("BatchDecode step %d: %v", i, err)
		}

		// Zero-copy path: SampleSeq reads logits directly in C++.
		tok, err := smpl.SampleSeq(seq)
		if err != nil {
			t.Fatalf("SampleSeq step %d: %v", i, err)
		}

		if m.IsEOG(tok) {
			break
		}

		piece, err := m.TokenToPiece(tok)
		if err != nil {
			t.Fatalf("TokenToPiece step %d: %v", i, err)
		}
		output = append(output, piece...)
		seq.AppendToken(tok)
	}

	t.Logf("grammar output: %s", string(output))

	// The output MUST be valid JSON (or empty if the model produced nothing).
	if len(output) == 0 {
		t.Skip("model produced no output")
	}
	if !json.Valid(output) {
		t.Errorf("grammar-constrained output is not valid JSON: %q", string(output))
	}
}

// ─── Custom GBNF grammar ────────────────────────────────────────────────────

func TestGrammarSamplerCustom(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 2, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	// Grammar that only allows "yes" or "no".
	yesNoGrammar := `root ::= ("yes" | "no")`

	smpl, err := llm.NewGrammarSampler(m, yesNoGrammar, "root", 0, 0.9, 0, 42)
	if err != nil {
		t.Fatalf("NewGrammarSampler: %v", err)
	}
	defer smpl.Close()

	tokens, err := m.Tokenize("Answer yes or no:", true, 64)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}

	seq, err := m.NewSequence(tokens)
	if err != nil {
		t.Fatalf("NewSequence: %v", err)
	}
	defer seq.Close()

	var output string
	for i := 0; i < 10; i++ {
		if err := m.BatchDecode([]*llm.Sequence{seq}); err != nil {
			t.Fatalf("BatchDecode step %d: %v", i, err)
		}
		tok, err := smpl.SampleSeq(seq)
		if err != nil {
			t.Fatalf("SampleSeq step %d: %v", i, err)
		}
		if m.IsEOG(tok) {
			break
		}
		piece, err := m.TokenToPiece(tok)
		if err != nil {
			t.Fatalf("TokenToPiece step %d: %v", i, err)
		}
		output += piece
		seq.AppendToken(tok)
	}

	t.Logf("yes/no grammar output: %q", output)
	if output != "yes" && output != "no" {
		t.Errorf("expected 'yes' or 'no', got %q", output)
	}
}
