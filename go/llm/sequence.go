package llm

/*
#include "infer_api.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
)

// Sequence wraps an InferSeq handle. It holds a KV cache slot and the
// full generation state for one concurrent inference sequence.
// Always call Close when done to release the KV cache slot.
type Sequence struct {
	ptr   C.InferSeq
	model *Model
}

// Close destroys the sequence and releases its KV cache slot.
// Safe to call multiple times.
func (s *Sequence) Close() {
	if s.ptr == nil {
		return
	}
	C.infer_seq_destroy(s.ptr)
	s.ptr = nil
	runtime.SetFinalizer(s, nil)
}

// IsDone returns true if the sequence has generated an end-of-generation token.
func (s *Sequence) IsDone() bool {
	if s.ptr == nil {
		return true
	}
	return C.infer_seq_is_done(s.ptr) != 0
}

// Position returns the current KV cache position (number of tokens decoded so far).
func (s *Sequence) Position() int {
	if s.ptr == nil {
		return 0
	}
	return int(C.infer_seq_position(s.ptr))
}

// SlotID returns the KV cache slot ID assigned to this sequence.
func (s *Sequence) SlotID() int {
	if s.ptr == nil {
		return -1
	}
	return int(C.infer_seq_slot_id(s.ptr))
}

// AppendToken adds a sampled token to the sequence history and advances the KV
// position. A no-op if the sequence is already done.
func (s *Sequence) AppendToken(id int32) {
	if s.ptr == nil {
		return
	}
	C.infer_seq_append_token(s.ptr, C.int(id))
}

// Logits returns the logit vector produced by the last BatchDecode call.
// Returns an error if BatchDecode has not been called yet for this sequence.
func (s *Sequence) Logits() ([]float32, error) {
	if s.ptr == nil {
		return nil, errors.New("llm: Logits called on closed sequence")
	}
	if s.model == nil || s.model.vocabSize == 0 {
		return nil, errors.New("llm: Logits: model not available")
	}

	vocab := s.model.vocabSize
	logits := make([]float32, vocab)
	rc := C.infer_seq_get_logits(s.ptr, (*C.float)(&logits[0]), C.int(vocab))
	if rc != C.INFER_OK {
		return nil, fmt.Errorf("llm: Logits failed: %w", lastError())
	}
	return logits, nil
}

// SampleToken samples the next token from the logit distribution produced by
// the last BatchDecode call.
// temperature: controls randomness (0 = greedy/argmax, 1.0 = default).
// topP: nucleus sampling threshold (0 or 1.0 = disabled).
// Returns the sampled token ID.
func (s *Sequence) SampleToken(temperature, topP float32) (int32, error) {
	logits, err := s.Logits()
	if err != nil {
		return 0, err
	}
	return sampleLogits(logits, temperature, topP), nil
}

// sampleLogits applies temperature scaling and optional top-p nucleus sampling.
func sampleLogits(logits []float32, temperature, topP float32) int32 {
	if temperature <= 0 {
		// Greedy: argmax
		best := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return int32(best)
	}

	// Temperature scaling + softmax (numerically stable)
	scaled := make([]float64, len(logits))
	maxVal := float64(logits[0])
	for _, l := range logits[1:] {
		if v := float64(l); v > maxVal {
			maxVal = v
		}
	}
	var sum float64
	for i, l := range logits {
		scaled[i] = math.Exp((float64(l) - maxVal) / float64(temperature))
		sum += scaled[i]
	}
	for i := range scaled {
		scaled[i] /= sum
	}

	// Top-p nucleus sampling
	if topP > 0 && topP < 1.0 {
		type tp struct {
			id   int
			prob float64
		}
		candidates := make([]tp, len(scaled))
		for i, p := range scaled {
			candidates[i] = tp{i, p}
		}
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].prob > candidates[j].prob
		})
		cumulative := 0.0
		cutoff := 0
		for cutoff < len(candidates) {
			cumulative += candidates[cutoff].prob
			cutoff++
			if cumulative >= float64(topP) {
				break
			}
		}
		candidates = candidates[:cutoff]

		var newSum float64
		for _, c := range candidates {
			newSum += c.prob
		}
		r := rand.Float64() * newSum
		acc := 0.0
		for _, c := range candidates {
			acc += c.prob
			if r <= acc {
				return int32(c.id)
			}
		}
		return int32(candidates[len(candidates)-1].id)
	}

	// Weighted sample over full distribution
	r := rand.Float64()
	acc := 0.0
	for i, p := range scaled {
		acc += p
		if r <= acc {
			return int32(i)
		}
	}
	return int32(len(scaled) - 1)
}
