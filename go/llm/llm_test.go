package llm_test

import (
	"os"
	"testing"

	"github.com/ailakshya/infergo/llm"
)

const testModelPath = "/tmp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

func skipIfMissing(t *testing.T) {
	t.Helper()
	if _, err := os.Stat(testModelPath); err != nil {
		t.Skipf("LLM model not found at %s", testModelPath)
	}
}

// ─── Load / Close ─────────────────────────────────────────────────────────────

func TestLoad(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 4, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	m.Close()
}

func TestLoadBadPath(t *testing.T) {
	_, err := llm.Load("/no/such/model.gguf", 0, 512, 4, 256)
	if err == nil {
		t.Fatal("expected error for bad path, got nil")
	}
}

func TestDoubleClose(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 4, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	m.Close()
	m.Close() // must not panic
}

// ─── VocabSize / BOS / EOS / IsEOG ───────────────────────────────────────────

func TestVocabSize(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 4, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if m.VocabSize() <= 0 {
		t.Errorf("VocabSize() = %d, want > 0", m.VocabSize())
	}
}

func TestBOSEOS(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 4, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if m.BOS() < 0 {
		t.Errorf("BOS() = %d, want >= 0", m.BOS())
	}
	if m.EOS() < 0 {
		t.Errorf("EOS() = %d, want >= 0", m.EOS())
	}
}

func TestIsEOG(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 4, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if !m.IsEOG(m.EOS()) {
		t.Errorf("IsEOG(EOS=%d) = false, want true", m.EOS())
	}
	if m.IsEOG(0) {
		t.Error("IsEOG(0) = true, want false")
	}
}

// ─── NewSequence ──────────────────────────────────────────────────────────────

func TestNewSequence(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 4, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	seq, err := m.NewSequence([]int32{m.BOS()})
	if err != nil {
		t.Fatalf("NewSequence: %v", err)
	}
	defer seq.Close()

	if seq.IsDone() {
		t.Error("fresh sequence should not be done")
	}
	if seq.SlotID() < 0 {
		t.Errorf("SlotID() = %d, want >= 0", seq.SlotID())
	}
	if seq.Position() != 0 {
		t.Errorf("Position() = %d before decode, want 0", seq.Position())
	}
}

func TestNewSequenceEmptyTokens(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 4, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	_, err = m.NewSequence(nil)
	if err == nil {
		t.Fatal("expected error for empty tokens, got nil")
	}
}

func TestSequenceDoubleClose(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 4, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	seq, err := m.NewSequence([]int32{m.BOS()})
	if err != nil {
		t.Fatalf("NewSequence: %v", err)
	}
	seq.Close()
	seq.Close() // must not panic
}

func TestFourSequencesDistinctSlots(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 1024, 4, 512)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	bos := m.BOS()
	seqs := make([]*llm.Sequence, 4)
	for i := range seqs {
		seqs[i], err = m.NewSequence([]int32{bos, int32(100 * (i + 1))})
		if err != nil {
			t.Fatalf("NewSequence[%d]: %v", i, err)
		}
		defer seqs[i].Close()
	}

	slots := make(map[int]bool)
	for i, s := range seqs {
		id := s.SlotID()
		if slots[id] {
			t.Errorf("seqs[%d] slot %d already used by another sequence", i, id)
		}
		slots[id] = true
	}
}

// ─── BatchDecode / Logits ─────────────────────────────────────────────────────

func TestBatchDecodeFourSequences(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 1024, 4, 512)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	bos := m.BOS()
	prompts := [][]int32{
		{bos},
		{bos, 100},
		{bos, 200, 300},
		{bos, 400, 500, 600},
	}

	seqs := make([]*llm.Sequence, len(prompts))
	for i, p := range prompts {
		seqs[i], err = m.NewSequence(p)
		if err != nil {
			t.Fatalf("NewSequence[%d]: %v", i, err)
		}
		defer seqs[i].Close()
	}

	if err := m.BatchDecode(seqs); err != nil {
		t.Fatalf("BatchDecode: %v", err)
	}

	for i, seq := range seqs {
		logits, err := seq.Logits()
		if err != nil {
			t.Errorf("seqs[%d].Logits(): %v", i, err)
			continue
		}
		if len(logits) != m.VocabSize() {
			t.Errorf("seqs[%d] logits len %d, want %d", i, len(logits), m.VocabSize())
		}
		var absSum float32
		for _, v := range logits {
			if v < 0 {
				absSum -= v
			} else {
				absSum += v
			}
		}
		if absSum == 0 {
			t.Errorf("seqs[%d] logits are all zero", i)
		}
	}
}

// ─── SampleToken ─────────────────────────────────────────────────────────────

func TestSampleTokenGreedyDeterministic(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 2, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	seq, err := m.NewSequence([]int32{m.BOS()})
	if err != nil {
		t.Fatalf("NewSequence: %v", err)
	}
	defer seq.Close()

	if err := m.BatchDecode([]*llm.Sequence{seq}); err != nil {
		t.Fatalf("BatchDecode: %v", err)
	}

	tok1, err := seq.SampleToken(0, 0)
	if err != nil {
		t.Fatalf("SampleToken: %v", err)
	}
	tok2, err := seq.SampleToken(0, 0)
	if err != nil {
		t.Fatalf("SampleToken: %v", err)
	}
	if tok1 != tok2 {
		t.Errorf("greedy sampling not deterministic: %d != %d", tok1, tok2)
	}
	if tok1 < 0 || int(tok1) >= m.VocabSize() {
		t.Errorf("sampled token %d out of range [0, %d)", tok1, m.VocabSize())
	}
}

// ─── AppendToken / IsDone / Position ─────────────────────────────────────────

func TestAppendTokenIsDone(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 2, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	seq, err := m.NewSequence([]int32{m.BOS()})
	if err != nil {
		t.Fatalf("NewSequence: %v", err)
	}
	defer seq.Close()

	if seq.IsDone() {
		t.Fatal("sequence should not be done before EOS")
	}
	seq.AppendToken(m.EOS())
	if !seq.IsDone() {
		t.Error("sequence should be done after EOS appended")
	}
}

// ─── Full generation loop ─────────────────────────────────────────────────────

func TestGenerationLoop(t *testing.T) {
	skipIfMissing(t)
	m, err := llm.Load(testModelPath, 99, 512, 2, 256)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()

	seq, err := m.NewSequence([]int32{m.BOS()})
	if err != nil {
		t.Fatalf("NewSequence: %v", err)
	}
	defer seq.Close()

	const maxSteps = 10
	steps := 0
	for !seq.IsDone() && steps < maxSteps {
		if err := m.BatchDecode([]*llm.Sequence{seq}); err != nil {
			t.Fatalf("BatchDecode step %d: %v", steps, err)
		}
		tok, err := seq.SampleToken(0, 0) // greedy
		if err != nil {
			t.Fatalf("SampleToken step %d: %v", steps, err)
		}
		seq.AppendToken(tok)
		steps++
	}

	if steps == 0 {
		t.Error("generation loop did not execute any steps")
	}
	if seq.Position() <= 1 {
		t.Errorf("position %d, expected > 1 after %d steps", seq.Position(), steps)
	}
}
