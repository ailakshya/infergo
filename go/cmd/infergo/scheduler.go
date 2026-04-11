package main

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/ailakshya/infergo/llm"
	"github.com/ailakshya/infergo/server"
	"github.com/prometheus/client_golang/prometheus"
)

// TokenEvent is one item emitted on a request's token channel.
// When the channel is closed, generation is complete.
type TokenEvent struct {
	Piece string // decoded text piece for this token
	Err   error  // non-nil: generation failed; channel will be closed after this
}

// schedRequest is a single enqueued generation request.
type schedRequest struct {
	ctx       context.Context
	tokens    []int32       // pre-tokenized prompt
	maxTokens int           // max generation tokens
	temp      float32       // sampling temperature
	grammar   string        // GBNF grammar constraint (empty = none)
	tokenCh   chan TokenEvent // scheduler writes here; closed when done
}

// activeSeq tracks one in-progress sequence inside the scheduler goroutine.
type activeSeq struct {
	req     *schedRequest
	seq     *llm.Sequence
	sampler *llm.Sampler // grammar-constrained sampler (nil = use Go-side sampling)
	gen     int          // tokens generated so far
}

// schedulerModel wraps an llm.Model with a continuous batching scheduler.
// All Generate and Stream calls are routed through a single goroutine that
// calls BatchDecode on all active sequences simultaneously — eliminating
// the per-request mutex and enabling true concurrent GPU utilization.
//
// Implements server.Model, server.LLMModel, and server.StreamingLLMModel.
type schedulerModel struct {
	m               *llm.Model
	modelName       string
	submitCh        chan *schedRequest
	stopCh          chan struct{}
	wg              sync.WaitGroup
	activeSeqGauge  prometheus.Gauge
	metrics         *server.Metrics
	maxBatchSize    int           // cap on sequences per BatchDecode call (0 = unlimited)
	batchTimeout    time.Duration // how long to wait for more requests after first arrives
	gcInterval      int           // call runtime.GC() every N completed requests (0 = disabled)
	completedReqs   int           // count of requests completed since last GC (scheduler goroutine only)
	specDecoder     *llm.SpeculativeDecoder // optional: speculative decoding engine
}

// newSchedulerModel creates and starts a schedulerModel for the given model.
// The scheduler goroutine runs until Close is called.
// activeSeqGauge may be nil; when non-nil it tracks the number of sequences
// currently decoding (incremented on initSeq, decremented on sequence close).
// metrics may be nil; when non-nil KV page gauges are updated after each
// BatchDecode call.
// maxBatchSize caps the number of sequences per BatchDecode call (0 = unlimited).
// batchTimeoutMs is how long (in milliseconds) to wait for additional requests
// to arrive after the first one before firing a batch (0 = no wait).
// gcInterval causes runtime.GC() to be called every N completed requests
// (0 = disabled). This bounds RSS growth under sustained sequential load.
func newSchedulerModel(m *llm.Model, modelName string, activeSeqGauge prometheus.Gauge, metrics *server.Metrics, maxBatchSize, batchTimeoutMs, gcInterval int) *schedulerModel {
	s := &schedulerModel{
		m:              m,
		modelName:      modelName,
		submitCh:       make(chan *schedRequest, 256),
		stopCh:         make(chan struct{}),
		activeSeqGauge: activeSeqGauge,
		metrics:        metrics,
		maxBatchSize:   maxBatchSize,
		batchTimeout:   time.Duration(batchTimeoutMs) * time.Millisecond,
		gcInterval:     gcInterval,
	}
	s.wg.Add(1)
	go s.run()
	return s
}

// Close stops the scheduler and releases C-side model resources.
// Implements server.Model.
func (s *schedulerModel) Close() {
	close(s.stopCh)
	s.wg.Wait()
	if s.specDecoder != nil {
		s.specDecoder.Close()
	}
	s.m.Close()
}

// Generate tokenizes the prompt, submits it to the scheduler, and collects
// all generated tokens into a single string. Implements server.LLMModel.
func (s *schedulerModel) Generate(ctx context.Context, prompt string, maxTokens int, temp float32) (string, int, int, error) {
	tokens, err := s.m.Tokenize(prompt, true, 4096)
	if err != nil {
		return "", 0, 0, fmt.Errorf("tokenize: %w", err)
	}
	promptToks := len(tokens)

	// Speculative decoding fast path: entire generate loop runs in C++.
	// Bypasses the per-token scheduler — one CGo call for the full generation.
	_, hasGrammar := server.GrammarFromContext(ctx)
	if s.specDecoder != nil && !hasGrammar {
		text, stats, err := s.specDecoder.Generate(tokens, maxTokens, temp)
		if err != nil {
			return "", promptToks, 0, fmt.Errorf("speculative: %w", err)
		}
		_ = stats // TODO: expose stats via metrics
		return text, promptToks, stats.Predicted, nil
	}

	grammar, _ := server.GrammarFromContext(ctx)
	tokenCh, err := s.enqueue(ctx, tokens, maxTokens, temp, grammar)
	if err != nil {
		return "", promptToks, 0, err
	}

	var sb strings.Builder
	genToks := 0
	for ev := range tokenCh {
		if ev.Err != nil {
			return sb.String(), promptToks, genToks, ev.Err
		}
		sb.WriteString(ev.Piece)
		genToks++
	}
	return sb.String(), promptToks, genToks, nil
}

// Stream tokenizes the prompt, submits it to the scheduler, and returns a
// channel that emits decoded token pieces one at a time.
// Implements server.StreamingLLMModel.
func (s *schedulerModel) Stream(ctx context.Context, prompt string, maxTokens int, temp float32) (<-chan string, error) {
	tokens, err := s.m.Tokenize(prompt, true, 4096)
	if err != nil {
		return nil, fmt.Errorf("tokenize: %w", err)
	}

	grammar, _ := server.GrammarFromContext(ctx)
	tokenCh, err := s.enqueue(ctx, tokens, maxTokens, temp, grammar)
	if err != nil {
		return nil, err
	}

	out := make(chan string, 64)
	go func() {
		defer close(out)
		for ev := range tokenCh {
			if ev.Err != nil {
				return
			}
			select {
			case out <- ev.Piece:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

// enqueue submits a pre-tokenized request to the scheduler.
// Returns the channel to read tokens from, or an error.
func (s *schedulerModel) enqueue(ctx context.Context, tokens []int32, maxTokens int, temp float32, grammar string) (<-chan TokenEvent, error) {
	if maxTokens <= 0 {
		maxTokens = 256
	}
	// Do NOT override temp=0 — zero means greedy (argmax) sampling,
	// which is O(N) vs O(N log N) for top-p. Callers that want stochastic
	// sampling must set temp > 0 explicitly.
	req := &schedRequest{
		ctx:       ctx,
		tokens:    tokens,
		maxTokens: maxTokens,
		temp:      temp,
		grammar:   grammar,
		tokenCh:   make(chan TokenEvent, 64),
	}
	select {
	case s.submitCh <- req:
		return req.tokenCh, nil
	case <-s.stopCh:
		return nil, errors.New("scheduler: stopped")
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// run is the scheduler's main goroutine. It owns all activeSeq values and is
// the only goroutine that calls BatchDecode — no locking needed.
func (s *schedulerModel) run() {
	defer s.wg.Done()
	var active []*activeSeq

	for {
		// If no active sequences, block until a request arrives or we stop.
		if len(active) == 0 {
			select {
			case req := <-s.submitCh:
				if a := s.initSeq(req); a != nil {
					active = append(active, a)
				}
			case <-s.stopCh:
				return
			}

			// After the first request arrives on an empty queue, wait up to
			// batchTimeout for more requests to land so they share the same
			// BatchDecode call (reduces P50 latency at moderate concurrency).
			if s.batchTimeout > 0 {
				deadline := time.NewTimer(s.batchTimeout)
			batchWait:
				for {
					// Stop gathering if we've hit the batch size cap.
					if s.maxBatchSize > 0 && len(active) >= s.maxBatchSize {
						if !deadline.Stop() {
							<-deadline.C
						}
						break batchWait
					}
					select {
					case req := <-s.submitCh:
						if a := s.initSeq(req); a != nil {
							active = append(active, a)
						}
					case <-deadline.C:
						break batchWait
					case <-s.stopCh:
						deadline.Stop()
						for _, a := range active {
							close(a.req.tokenCh)
							s.closeActiveSeq(a)
						}
						return
					}
				}
			}
		}

		// Drain all pending requests without blocking (add to current batch),
		// respecting the maxBatchSize cap.
	drain:
		for {
			if s.maxBatchSize > 0 && len(active) >= s.maxBatchSize {
				break drain
			}
			select {
			case req := <-s.submitCh:
				if a := s.initSeq(req); a != nil {
					active = append(active, a)
				}
			default:
				break drain
			}
		}

		// Remove sequences whose client context was cancelled before decode.
		active = s.pruneCancelled(active)
		if len(active) == 0 {
			continue
		}

		// One BatchDecode call for all active sequences.
		seqs := make([]*llm.Sequence, len(active))
		for i, a := range active {
			seqs[i] = a.seq
		}
		if err := s.m.BatchDecode(seqs); err != nil {
			// Fatal decode error — report to all waiters and reset.
			for _, a := range active {
				a.req.tokenCh <- TokenEvent{Err: err}
				close(a.req.tokenCh)
				s.closeActiveSeq(a)
			}
			active = active[:0]
			continue
		}
		if s.metrics != nil {
			s.metrics.UpdateKVPages(s.modelName, s.m.KVPagesFree(), s.m.KVPagesTotal())
		}

		// Sample one token per sequence and route it.
		var next []*activeSeq
		for _, a := range active {
			var tok int32
			var err error
			if a.sampler != nil {
				// Grammar-constrained sampling: zero-copy path.
				// Reads logits directly from C++ SeqHandle — no data crosses CGo.
				tok, err = a.sampler.SampleSeq(a.seq)
			} else {
				// Standard Go-side sampling.
				tok, err = a.seq.SampleToken(a.req.temp, 0.9)
			}
			if err != nil {
				a.req.tokenCh <- TokenEvent{Err: err}
				close(a.req.tokenCh)
				s.closeActiveSeq(a)
				continue
			}

			// End of generation: EOG token or max_tokens reached.
			if s.m.IsEOG(tok) || a.gen >= a.req.maxTokens {
				close(a.req.tokenCh)
				s.closeActiveSeq(a)
				s.maybeGC()
				continue
			}

			piece, err := s.m.TokenToPiece(tok)
			if err != nil {
				a.req.tokenCh <- TokenEvent{Err: err}
				close(a.req.tokenCh)
				s.closeActiveSeq(a)
				continue
			}

			a.seq.AppendToken(tok)
			a.gen++

			// Deliver the token. If the client cancelled, discard and clean up.
			select {
			case a.req.tokenCh <- TokenEvent{Piece: piece}:
				next = append(next, a)
			case <-a.req.ctx.Done():
				close(a.req.tokenCh)
				s.closeActiveSeq(a)
			}
		}
		active = next

		// Honour stop signal between decode steps.
		select {
		case <-s.stopCh:
			for _, a := range active {
				close(a.req.tokenCh)
				s.closeActiveSeq(a)
			}
			return
		default:
		}
	}
}

// initSeq creates a new llm.Sequence for the request.
// Returns nil if the context is already done or sequence creation fails;
// in both cases tokenCh is closed with an appropriate error.
func (s *schedulerModel) initSeq(req *schedRequest) *activeSeq {
	select {
	case <-req.ctx.Done():
		req.tokenCh <- TokenEvent{Err: req.ctx.Err()}
		close(req.tokenCh)
		return nil
	default:
	}

	// Check if enough KV pages are free (need at least 1 page = 16 tokens minimum).
	free := s.m.KVPagesFree()
	if free <= 0 {
		req.tokenCh <- TokenEvent{Err: errors.New("KV cache exhausted: no pages free")}
		close(req.tokenCh)
		return nil
	}

	seq, err := s.m.NewSequence(req.tokens)
	if err != nil {
		req.tokenCh <- TokenEvent{Err: fmt.Errorf("new sequence: %w", err)}
		close(req.tokenCh)
		return nil
	}

	// Create grammar-constrained sampler if requested.
	var sampler *llm.Sampler
	if req.grammar != "" {
		sampler, err = llm.NewGrammarSampler(s.m, req.grammar, "root", req.temp, 0.9, 0, 0)
		if err != nil {
			seq.Close()
			req.tokenCh <- TokenEvent{Err: fmt.Errorf("grammar sampler: %w", err)}
			close(req.tokenCh)
			return nil
		}
	}

	if s.activeSeqGauge != nil {
		s.activeSeqGauge.Inc()
	}
	return &activeSeq{req: req, seq: seq, sampler: sampler}
}

// decActiveSeq decrements the active-sequences gauge if one is configured.
func (s *schedulerModel) decActiveSeq() {
	if s.activeSeqGauge != nil {
		s.activeSeqGauge.Dec()
	}
}

// closeActiveSeq closes the sequence, its grammar sampler (if any), and
// decrements the active gauge. Centralises cleanup to avoid leaking samplers.
func (s *schedulerModel) closeActiveSeq(a *activeSeq) {
	if a.sampler != nil {
		a.sampler.Close()
	}
	a.seq.Close()
	s.decActiveSeq()
}

// maybeGC increments the completed-request counter and calls runtime.GC()
// when the counter reaches gcInterval. This must only be called from the
// scheduler goroutine (no locking needed — completedReqs is owned by run).
func (s *schedulerModel) maybeGC() {
	if s.gcInterval <= 0 {
		return
	}
	s.completedReqs++
	if s.completedReqs >= s.gcInterval {
		s.completedReqs = 0
		runtime.GC()
	}
}

// PrefillPrompt tokenizes the prompt, runs a prefill-only forward pass through
// the model, serializes the resulting KV cache, and returns the bytes plus the
// prompt token count. Implements server.KVSerializable.
//
// This call runs entirely outside the continuous-batching scheduler so it does
// not interfere with normal generation traffic. The sequence is closed
// immediately after the KV cache is captured.
func (s *schedulerModel) PrefillPrompt(ctx context.Context, prompt string) ([]byte, int, error) {
	tokens, err := s.m.Tokenize(prompt, true, 4096)
	if err != nil {
		return nil, 0, fmt.Errorf("prefill: tokenize: %w", err)
	}
	if len(tokens) == 0 {
		return nil, 0, errors.New("prefill: empty prompt after tokenization")
	}

	seq, err := s.m.NewSequence(tokens)
	if err != nil {
		return nil, 0, fmt.Errorf("prefill: new sequence: %w", err)
	}
	defer seq.Close()

	// Run one BatchDecode to process all prompt tokens (prefill pass).
	if err := s.m.BatchDecode([]*llm.Sequence{seq}); err != nil {
		return nil, 0, fmt.Errorf("prefill: batch decode: %w", err)
	}

	// Serialize the KV cache for this sequence slot.
	kvBytes, err := s.m.SerializeKV(seq.SlotID())
	if err != nil {
		return nil, 0, fmt.Errorf("prefill: serialize KV: %w", err)
	}

	return kvBytes, len(tokens), nil
}

// DecodeFromKV deserializes KV cache bytes into a fresh sequence slot and runs
// generation for up to maxTokens steps, returning the generated text.
// Implements server.KVSerializable.
//
// Like PrefillPrompt, this bypasses the scheduler and runs synchronously.
func (s *schedulerModel) DecodeFromKV(ctx context.Context, kvBytes []byte, nPromptToks int, maxTokens int, temp float32) (string, error) {
	if len(kvBytes) == 0 {
		return "", errors.New("decode: kvBytes is empty")
	}
	if maxTokens <= 0 {
		maxTokens = 256
	}

	// Allocate a sequence slot with a single placeholder token.
	// The slot is needed to reserve a KV cache page; the actual state will be
	// overwritten by DeserializeKV immediately after.
	bos := s.m.BOS()
	if bos < 0 {
		bos = 1 // fallback
	}
	seq, err := s.m.NewSequence([]int32{bos})
	if err != nil {
		return "", fmt.Errorf("decode: new sequence: %w", err)
	}
	defer seq.Close()

	// Overwrite the slot's KV state with the serialized prefill result.
	if err := s.m.DeserializeKV(seq.SlotID(), kvBytes); err != nil {
		return "", fmt.Errorf("decode: deserialize KV: %w", err)
	}

	// Generation loop.
	var sb strings.Builder
	for i := 0; i < maxTokens; i++ {
		select {
		case <-ctx.Done():
			return sb.String(), ctx.Err()
		default:
		}

		if err := s.m.BatchDecode([]*llm.Sequence{seq}); err != nil {
			return sb.String(), fmt.Errorf("decode: batch decode step %d: %w", i, err)
		}

		tok, err := seq.SampleToken(temp, 0.9)
		if err != nil {
			return sb.String(), fmt.Errorf("decode: sample step %d: %w", i, err)
		}

		if s.m.IsEOG(tok) {
			break
		}

		piece, err := s.m.TokenToPiece(tok)
		if err != nil {
			return sb.String(), fmt.Errorf("decode: token-to-piece step %d: %w", i, err)
		}

		seq.AppendToken(tok)
		sb.WriteString(piece)
	}

	return sb.String(), nil
}

// pruneCancelled removes sequences whose client context has been cancelled,
// freeing their KV cache slots immediately.
func (s *schedulerModel) pruneCancelled(active []*activeSeq) []*activeSeq {
	out := active[:0]
	for _, a := range active {
		select {
		case <-a.req.ctx.Done():
			close(a.req.tokenCh)
			s.closeActiveSeq(a)
		default:
			out = append(out, a)
		}
	}
	return out
}
