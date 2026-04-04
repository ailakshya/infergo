package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/ailakshya/infergo/llm"
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
	tokenCh   chan TokenEvent // scheduler writes here; closed when done
}

// activeSeq tracks one in-progress sequence inside the scheduler goroutine.
type activeSeq struct {
	req *schedRequest
	seq *llm.Sequence
	gen int // tokens generated so far
}

// schedulerModel wraps an llm.Model with a continuous batching scheduler.
// All Generate and Stream calls are routed through a single goroutine that
// calls BatchDecode on all active sequences simultaneously — eliminating
// the per-request mutex and enabling true concurrent GPU utilization.
//
// Implements server.Model, server.LLMModel, and server.StreamingLLMModel.
type schedulerModel struct {
	m               *llm.Model
	submitCh        chan *schedRequest
	stopCh          chan struct{}
	wg              sync.WaitGroup
	activeSeqGauge  prometheus.Gauge
}

// newSchedulerModel creates and starts a schedulerModel for the given model.
// The scheduler goroutine runs until Close is called.
// activeSeqGauge may be nil; when non-nil it tracks the number of sequences
// currently decoding (incremented on initSeq, decremented on sequence close).
func newSchedulerModel(m *llm.Model, activeSeqGauge prometheus.Gauge) *schedulerModel {
	s := &schedulerModel{
		m:              m,
		submitCh:       make(chan *schedRequest, 256),
		stopCh:         make(chan struct{}),
		activeSeqGauge: activeSeqGauge,
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

	tokenCh, err := s.enqueue(ctx, tokens, maxTokens, temp)
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

	tokenCh, err := s.enqueue(ctx, tokens, maxTokens, temp)
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
func (s *schedulerModel) enqueue(ctx context.Context, tokens []int32, maxTokens int, temp float32) (<-chan TokenEvent, error) {
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
		}

		// Drain all pending requests without blocking (add to current batch).
	drain:
		for {
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
				a.seq.Close()
				s.decActiveSeq()
			}
			active = active[:0]
			continue
		}

		// Sample one token per sequence and route it.
		var next []*activeSeq
		for _, a := range active {
			tok, err := a.seq.SampleToken(a.req.temp, 0.9)
			if err != nil {
				a.req.tokenCh <- TokenEvent{Err: err}
				close(a.req.tokenCh)
				a.seq.Close()
				s.decActiveSeq()
				continue
			}

			// End of generation: EOG token or max_tokens reached.
			if s.m.IsEOG(tok) || a.gen >= a.req.maxTokens {
				close(a.req.tokenCh)
				a.seq.Close()
				s.decActiveSeq()
				continue
			}

			piece, err := s.m.TokenToPiece(tok)
			if err != nil {
				a.req.tokenCh <- TokenEvent{Err: err}
				close(a.req.tokenCh)
				a.seq.Close()
				s.decActiveSeq()
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
				a.seq.Close()
				s.decActiveSeq()
			}
		}
		active = next

		// Honour stop signal between decode steps.
		select {
		case <-s.stopCh:
			for _, a := range active {
				close(a.req.tokenCh)
				a.seq.Close()
				s.decActiveSeq()
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

	seq, err := s.m.NewSequence(req.tokens)
	if err != nil {
		req.tokenCh <- TokenEvent{Err: fmt.Errorf("new sequence: %w", err)}
		close(req.tokenCh)
		return nil
	}
	if s.activeSeqGauge != nil {
		s.activeSeqGauge.Inc()
	}
	return &activeSeq{req: req, seq: seq}
}

// decActiveSeq decrements the active-sequences gauge if one is configured.
func (s *schedulerModel) decActiveSeq() {
	if s.activeSeqGauge != nil {
		s.activeSeqGauge.Dec()
	}
}

// pruneCancelled removes sequences whose client context has been cancelled,
// freeing their KV cache slots immediately.
func (s *schedulerModel) pruneCancelled(active []*activeSeq) []*activeSeq {
	out := active[:0]
	for _, a := range active {
		select {
		case <-a.req.ctx.Done():
			close(a.req.tokenCh)
			a.seq.Close()
			s.decActiveSeq()
		default:
			out = append(out, a)
		}
	}
	return out
}
