package server

import (
	"fmt"
	"sync"
	"time"
)

// CostTracker estimates compute cost per request. OPT-108.
type CostTracker struct {
	mu            sync.Mutex
	costPer1kTok  map[string]float64 // model → cost per 1K tokens
	perKey        map[string]*costEntry
}

type costEntry struct {
	totalTokens int64
	totalCost   float64
	requests    int64
	lastUpdate  time.Time
}

// NewCostTracker creates a cost tracker.
func NewCostTracker() *CostTracker {
	return &CostTracker{
		costPer1kTok: map[string]float64{
			"default": 0.002, // $0.002 per 1K tokens
		},
		perKey: make(map[string]*costEntry),
	}
}

// SetRate sets the cost rate for a model.
func (ct *CostTracker) SetRate(model string, costPer1k float64) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	ct.costPer1kTok[model] = costPer1k
}

// Track records token usage and returns the cost.
func (ct *CostTracker) Track(apiKey, model string, tokens int) float64 {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	rate, ok := ct.costPer1kTok[model]
	if !ok {
		rate = ct.costPer1kTok["default"]
	}
	cost := rate * float64(tokens) / 1000.0

	entry, ok := ct.perKey[apiKey]
	if !ok {
		entry = &costEntry{}
		ct.perKey[apiKey] = entry
	}
	entry.totalTokens += int64(tokens)
	entry.totalCost += cost
	entry.requests++
	entry.lastUpdate = time.Now()

	return cost
}

// CostHeader returns the X-Compute-Cost header value.
func (ct *CostTracker) CostHeader(model string, tokens int) string {
	ct.mu.Lock()
	rate, ok := ct.costPer1kTok[model]
	ct.mu.Unlock()
	if !ok {
		rate = 0.002
	}
	cost := rate * float64(tokens) / 1000.0
	return fmt.Sprintf("$%.6f", cost)
}

// Usage returns total usage for an API key.
func (ct *CostTracker) Usage(apiKey string) (tokens int64, cost float64, requests int64) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	entry, ok := ct.perKey[apiKey]
	if !ok {
		return 0, 0, 0
	}
	return entry.totalTokens, entry.totalCost, entry.requests
}
