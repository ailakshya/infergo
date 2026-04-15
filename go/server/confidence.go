package server

import "math"

// ComputeConfidence estimates model confidence from token log probabilities.
// Higher average logprob = more confident output.
// Returns a score in [0, 1].
func ComputeConfidence(logprobs []float64) float64 {
	if len(logprobs) == 0 {
		return 0.5 // no data = uncertain
	}

	// Average log probability
	sum := 0.0
	for _, lp := range logprobs {
		sum += lp
	}
	avgLogprob := sum / float64(len(logprobs))

	// Convert log probability to confidence score [0, 1]
	// logprob = 0 means probability = 1 (max confidence)
	// logprob = -inf means probability = 0 (no confidence)
	// Map [-10, 0] → [0, 1] with sigmoid-like curve
	confidence := 1.0 / (1.0 + math.Exp(-2.0*(avgLogprob+2.0)))

	if confidence < 0 {
		confidence = 0
	}
	if confidence > 1 {
		confidence = 1
	}

	return confidence
}
