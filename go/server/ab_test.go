package server

import (
	"math/rand"
	"sync"
	"sync/atomic"
)

// ABTest routes traffic between two models for comparison.
type ABTest struct {
	mu       sync.RWMutex
	modelA   string
	modelB   string
	splitPct float64 // percentage of traffic to model B (0.0-1.0)
	countA   atomic.Int64
	countB   atomic.Int64
	active   bool
}

// NewABTest creates an A/B test config.
func NewABTest(modelA, modelB string, splitPct float64) *ABTest {
	return &ABTest{
		modelA:   modelA,
		modelB:   modelB,
		splitPct: splitPct,
		active:   true,
	}
}

// Route returns which model to use for this request.
// Returns (model_name, variant "A" or "B").
func (ab *ABTest) Route(requestModel string) (string, string) {
	ab.mu.RLock()
	defer ab.mu.RUnlock()

	if !ab.active || requestModel != ab.modelA {
		return requestModel, ""
	}

	if rand.Float64() < ab.splitPct {
		ab.countB.Add(1)
		return ab.modelB, "B"
	}
	ab.countA.Add(1)
	return ab.modelA, "A"
}

// Stats returns traffic counts.
func (ab *ABTest) Stats() (a, b int64) {
	return ab.countA.Load(), ab.countB.Load()
}

// Disable stops the A/B test (all traffic to model A).
func (ab *ABTest) Disable() {
	ab.mu.Lock()
	defer ab.mu.Unlock()
	ab.active = false
}
