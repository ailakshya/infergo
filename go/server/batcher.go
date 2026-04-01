// Package server provides serving infrastructure for infergo models.
package server

import (
	"errors"
	"sync"
	"time"

	"github.com/ailakshya/infergo/tensor"
)

// BatchRequest is a single inference request submitted to a BatchScheduler.
type BatchRequest struct {
	Input   *tensor.Tensor
	replyCh chan BatchResponse
}

// BatchResponse is the result returned to the caller of Submit.
type BatchResponse struct {
	Output *tensor.Tensor
	Err    error
}

// ProcessFn is the user-supplied batch inference function.
// It receives a slice of N input tensors and must return exactly N output
// tensors in the same order, or a non-nil error.
type ProcessFn func(inputs []*tensor.Tensor) ([]*tensor.Tensor, error)

// BatchScheduler collects individual inference requests and dispatches them
// in batches to a ProcessFn. A batch is dispatched when either:
//   - MaxBatch requests have accumulated, or
//   - MaxWait has elapsed since the first request arrived in the batch.
//
// Use New to create and start a scheduler; call Stop when done.
type BatchScheduler struct {
	in       chan *BatchRequest
	maxBatch int
	maxWait  time.Duration
	process  ProcessFn
	stop     chan struct{}
	wg       sync.WaitGroup
}

// New creates and starts a BatchScheduler with the given parameters.
// maxBatch must be >= 1; maxWait must be > 0.
func New(maxBatch int, maxWait time.Duration, process ProcessFn) (*BatchScheduler, error) {
	if maxBatch < 1 {
		return nil, errors.New("batcher: maxBatch must be >= 1")
	}
	if maxWait <= 0 {
		return nil, errors.New("batcher: maxWait must be positive")
	}
	if process == nil {
		return nil, errors.New("batcher: process function must not be nil")
	}
	s := &BatchScheduler{
		in:       make(chan *BatchRequest, maxBatch*4),
		maxBatch: maxBatch,
		maxWait:  maxWait,
		process:  process,
		stop:     make(chan struct{}),
	}
	s.wg.Add(1)
	go s.run()
	return s, nil
}

// Submit enqueues a request and blocks until the batch result is available.
// Returns an error if the scheduler has been stopped.
func (s *BatchScheduler) Submit(req *tensor.Tensor) (*tensor.Tensor, error) {
	r := &BatchRequest{
		Input:   req,
		replyCh: make(chan BatchResponse, 1),
	}
	select {
	case s.in <- r:
	case <-s.stop:
		return nil, errors.New("batcher: scheduler stopped")
	}
	resp := <-r.replyCh
	return resp.Output, resp.Err
}

// Stop signals the scheduler to drain pending requests and shut down.
// Blocks until the worker goroutine exits.
func (s *BatchScheduler) Stop() {
	close(s.stop)
	s.wg.Wait()
}

// run is the scheduler's worker goroutine.
func (s *BatchScheduler) run() {
	defer s.wg.Done()

	for {
		// Wait for the first request or shutdown.
		var first *BatchRequest
		select {
		case first = <-s.in:
		case <-s.stop:
			s.drainWith(errors.New("batcher: scheduler stopped"))
			return
		}

		// Accumulate up to maxBatch within maxWait.
		batch := make([]*BatchRequest, 0, s.maxBatch)
		batch = append(batch, first)
		timer := time.NewTimer(s.maxWait)

	fill:
		for len(batch) < s.maxBatch {
			select {
			case r := <-s.in:
				batch = append(batch, r)
			case <-timer.C:
				break fill
			case <-s.stop:
				timer.Stop()
				s.dispatch(batch)
				s.drainWith(errors.New("batcher: scheduler stopped"))
				return
			}
		}
		timer.Stop()

		s.dispatch(batch)
	}
}

// dispatch calls the ProcessFn and fans results back to each requester.
func (s *BatchScheduler) dispatch(batch []*BatchRequest) {
	inputs := make([]*tensor.Tensor, len(batch))
	for i, r := range batch {
		inputs[i] = r.Input
	}

	outputs, err := s.process(inputs)

	for i, r := range batch {
		if err != nil {
			r.replyCh <- BatchResponse{Err: err}
			continue
		}
		if i >= len(outputs) {
			r.replyCh <- BatchResponse{Err: errors.New("batcher: process returned fewer outputs than inputs")}
			continue
		}
		r.replyCh <- BatchResponse{Output: outputs[i]}
	}
}

// drainWith empties the input channel and replies with err to all pending requests.
func (s *BatchScheduler) drainWith(err error) {
	for {
		select {
		case r := <-s.in:
			r.replyCh <- BatchResponse{Err: err}
		default:
			return
		}
	}
}
