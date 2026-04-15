package server

import (
	"math"
	"sync"
)

// DriftDetector tracks output distribution changes over time. OPT-110.
type DriftDetector struct {
	mu        sync.Mutex
	baseline  []float64 // centroid embedding of baseline outputs
	window    [][]float64 // recent output embeddings
	windowMax int
	threshold float64 // cosine distance threshold for drift alert
}

// NewDriftDetector creates a drift detector.
func NewDriftDetector(threshold float64, windowSize int) *DriftDetector {
	if threshold <= 0 {
		threshold = 0.15
	}
	if windowSize <= 0 {
		windowSize = 100
	}
	return &DriftDetector{
		windowMax: windowSize,
		threshold: threshold,
	}
}

// AddSample adds an output embedding to the rolling window.
func (d *DriftDetector) AddSample(embedding []float64) {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.window = append(d.window, embedding)
	if len(d.window) > d.windowMax {
		d.window = d.window[1:]
	}

	// Set baseline from first full window
	if d.baseline == nil && len(d.window) >= d.windowMax {
		d.baseline = centroid(d.window)
	}
}

// CheckDrift returns (drifted, distance) comparing current window to baseline.
func (d *DriftDetector) CheckDrift() (bool, float64) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.baseline == nil || len(d.window) < 10 {
		return false, 0
	}

	current := centroid(d.window)
	dist := cosineDistance(d.baseline, current)
	return dist > d.threshold, dist
}

func centroid(vecs [][]float64) []float64 {
	if len(vecs) == 0 {
		return nil
	}
	dim := len(vecs[0])
	result := make([]float64, dim)
	for _, v := range vecs {
		for i := range result {
			if i < len(v) {
				result[i] += v[i]
			}
		}
	}
	n := float64(len(vecs))
	for i := range result {
		result[i] /= n
	}
	return result
}

func cosineDistance(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 1.0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 1.0
	}
	sim := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	return 1.0 - sim
}
