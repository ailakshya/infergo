package tokenizer_test

import (
	"os"
	"testing"

	"github.com/ailakshya/infergo/tokenizer"
)

const testTokenizerPath = "/tmp/bert_tokenizer/tokenizer.json"

func skipIfMissing(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(testTokenizerPath); err != nil {
		t.Skipf("tokenizer not found at %s — run: python3 -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased').save_pretrained('/tmp/bert_tokenizer')\"", testTokenizerPath)
	}
}

// ─── Load / Close ─────────────────────────────────────────────────────────────

func TestLoad(t *testing.T) {
	skipIfMissing(t)
	tok, err := tokenizer.Load(testTokenizerPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	tok.Close()
}

func TestLoadBadPath(t *testing.T) {
	_, err := tokenizer.Load("/no/such/tokenizer.json")
	if err == nil {
		t.Fatal("expected error for bad path, got nil")
	}
}

func TestDoubleClose(t *testing.T) {
	skipIfMissing(t)
	tok, err := tokenizer.Load(testTokenizerPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	tok.Close()
	tok.Close() // must not panic
}

// ─── VocabSize ────────────────────────────────────────────────────────────────

func TestVocabSize(t *testing.T) {
	skipIfMissing(t)
	tok, err := tokenizer.Load(testTokenizerPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer tok.Close()

	if tok.VocabSize() <= 0 {
		t.Errorf("VocabSize() = %d, want > 0", tok.VocabSize())
	}
}

// ─── Encode ───────────────────────────────────────────────────────────────────

func TestEncode(t *testing.T) {
	skipIfMissing(t)
	tok, err := tokenizer.Load(testTokenizerPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer tok.Close()

	enc, err := tok.Encode("Hello world", false, 512)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(enc.IDs) == 0 {
		t.Error("Encode returned empty IDs")
	}
	if len(enc.IDs) != len(enc.AttentionMask) {
		t.Errorf("IDs len %d != AttentionMask len %d", len(enc.IDs), len(enc.AttentionMask))
	}
	for i, m := range enc.AttentionMask {
		if m != 1 {
			t.Errorf("attention_mask[%d] = %d, want 1", i, m)
		}
	}
}

func TestEncodeWithSpecialTokens(t *testing.T) {
	skipIfMissing(t)
	tok, err := tokenizer.Load(testTokenizerPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer tok.Close()

	without, err := tok.Encode("Hello", false, 512)
	if err != nil {
		t.Fatalf("Encode without special: %v", err)
	}
	with, err := tok.Encode("Hello", true, 512)
	if err != nil {
		t.Fatalf("Encode with special: %v", err)
	}
	if len(with.IDs) < len(without.IDs) {
		t.Errorf("with special tokens (%d) shorter than without (%d)", len(with.IDs), len(without.IDs))
	}
}

func TestEncodeTruncates(t *testing.T) {
	skipIfMissing(t)
	tok, err := tokenizer.Load(testTokenizerPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer tok.Close()

	enc, err := tok.Encode("the quick brown fox jumps over the lazy dog", false, 3)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(enc.IDs) != 3 {
		t.Errorf("want 3 tokens, got %d", len(enc.IDs))
	}
}

func TestEncodeEmpty(t *testing.T) {
	skipIfMissing(t)
	tok, err := tokenizer.Load(testTokenizerPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer tok.Close()

	_, err = tok.Encode("", false, 512)
	if err != nil {
		t.Errorf("Encode empty string: %v", err)
	}
}

// ─── Decode ───────────────────────────────────────────────────────────────────

func TestDecodeRoundTrip(t *testing.T) {
	skipIfMissing(t)
	tok, err := tokenizer.Load(testTokenizerPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer tok.Close()

	enc, err := tok.Encode("hello world", false, 512)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	text, err := tok.Decode(enc.IDs, true)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if text != "hello world" {
		t.Errorf("round-trip: got %q, want %q", text, "hello world")
	}
}

func TestDecodeEmpty(t *testing.T) {
	skipIfMissing(t)
	tok, err := tokenizer.Load(testTokenizerPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer tok.Close()

	text, err := tok.Decode(nil, true)
	if err != nil {
		t.Fatalf("Decode nil: %v", err)
	}
	if text != "" {
		t.Errorf("Decode nil: got %q, want empty string", text)
	}
}

func TestDecodeMultipleSentences(t *testing.T) {
	skipIfMissing(t)
	tok, err := tokenizer.Load(testTokenizerPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer tok.Close()

	sentences := []string{
		"hello world",
		"the quick brown fox",
		"inference at the speed of go",
	}
	for _, s := range sentences {
		enc, err := tok.Encode(s, false, 512)
		if err != nil {
			t.Fatalf("Encode %q: %v", s, err)
		}
		got, err := tok.Decode(enc.IDs, true)
		if err != nil {
			t.Fatalf("Decode %q: %v", s, err)
		}
		if got != s {
			t.Errorf("round-trip %q: got %q", s, got)
		}
	}
}

// ─── DecodeToken ──────────────────────────────────────────────────────────────

func TestDecodeToken(t *testing.T) {
	skipIfMissing(t)
	tok, err := tokenizer.Load(testTokenizerPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer tok.Close()

	enc, err := tok.Encode("hello", false, 512)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	for _, id := range enc.IDs {
		piece, err := tok.DecodeToken(id)
		if err != nil {
			t.Fatalf("DecodeToken(%d): %v", id, err)
		}
		if piece == "" {
			t.Errorf("DecodeToken(%d) returned empty string", id)
		}
	}
}
