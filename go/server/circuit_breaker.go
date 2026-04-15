package server

import (
	"sync"
	"time"
)

// CircuitBreaker tracks failures per model and auto-disables after threshold.
// States: closed (normal) → open (disabled) → half-open (testing).
type CircuitBreaker struct {
	mu       sync.RWMutex
	circuits map[string]*circuit
	threshold int           // failures before opening
	cooldown  time.Duration // time before half-open
}

type circuit struct {
	failures    int
	state       string // "closed", "open", "half-open"
	lastFailure time.Time
	lastSuccess time.Time
}

// NewCircuitBreaker creates a circuit breaker.
// threshold: number of consecutive failures before opening.
// cooldownSec: seconds to wait before trying half-open.
func NewCircuitBreaker(threshold, cooldownSec int) *CircuitBreaker {
	if threshold <= 0 {
		threshold = 5
	}
	if cooldownSec <= 0 {
		cooldownSec = 30
	}
	return &CircuitBreaker{
		circuits:  make(map[string]*circuit),
		threshold: threshold,
		cooldown:  time.Duration(cooldownSec) * time.Second,
	}
}

// Allow checks if a request to the model should be allowed.
// Returns true if circuit is closed or half-open (test request).
func (cb *CircuitBreaker) Allow(model string) bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	c, ok := cb.circuits[model]
	if !ok {
		return true // no circuit = allow
	}

	switch c.state {
	case "closed":
		return true
	case "open":
		if time.Since(c.lastFailure) > cb.cooldown {
			c.state = "half-open"
			return true // allow one test request
		}
		return false
	case "half-open":
		return false // already testing, block others
	}
	return true
}

// RecordSuccess records a successful request. Resets circuit to closed.
func (cb *CircuitBreaker) RecordSuccess(model string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	c, ok := cb.circuits[model]
	if !ok {
		return
	}
	c.failures = 0
	c.state = "closed"
	c.lastSuccess = time.Now()
}

// RecordFailure records a failed request. Opens circuit after threshold.
func (cb *CircuitBreaker) RecordFailure(model string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	c, ok := cb.circuits[model]
	if !ok {
		c = &circuit{state: "closed"}
		cb.circuits[model] = c
	}

	c.failures++
	c.lastFailure = time.Now()

	if c.failures >= cb.threshold {
		c.state = "open"
	}
}

// State returns the current state for a model.
func (cb *CircuitBreaker) State(model string) string {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	c, ok := cb.circuits[model]
	if !ok {
		return "closed"
	}
	return c.state
}
