package server

import (
	"strings"
)

// HallucinationChecker verifies LLM claims against source documents. OPT-124.
type HallucinationChecker struct {
	threshold float64 // minimum word overlap ratio to consider "supported"
}

// NewHallucinationChecker creates a hallucination checker.
func NewHallucinationChecker(threshold float64) *HallucinationChecker {
	if threshold <= 0 {
		threshold = 0.3 // 30% word overlap = supported
	}
	return &HallucinationChecker{threshold: threshold}
}

// VerificationResult holds the result of checking an LLM response.
type VerificationResult struct {
	Verified           bool     `json:"verified"`
	SupportedClaims    []string `json:"supported_claims,omitempty"`
	UnsupportedClaims  []string `json:"unsupported_claims,omitempty"`
	SupportRatio       float64  `json:"support_ratio"`
}

// Verify checks if the response is supported by the source documents.
// Splits response into sentences and checks each against sources.
func (hc *HallucinationChecker) Verify(response string, sources []string) VerificationResult {
	// Join all sources into one searchable corpus
	corpus := strings.ToLower(strings.Join(sources, " "))
	corpusWords := wordSet(corpus)

	// Split response into sentences
	sentences := splitSentences(response)

	var supported, unsupported []string
	for _, sent := range sentences {
		sent = strings.TrimSpace(sent)
		if len(sent) < 10 { // skip very short fragments
			continue
		}

		sentWords := wordSet(strings.ToLower(sent))
		overlap := wordOverlap(sentWords, corpusWords)

		if overlap >= hc.threshold {
			supported = append(supported, sent)
		} else {
			unsupported = append(unsupported, sent)
		}
	}

	total := len(supported) + len(unsupported)
	ratio := 0.0
	if total > 0 {
		ratio = float64(len(supported)) / float64(total)
	}

	return VerificationResult{
		Verified:          ratio >= 0.5,
		SupportedClaims:   supported,
		UnsupportedClaims: unsupported,
		SupportRatio:      ratio,
	}
}

func splitSentences(text string) []string {
	// Simple sentence splitter
	var sentences []string
	for _, delim := range []string{". ", "! ", "? ", ".\n", "!\n", "?\n"} {
		parts := strings.Split(text, delim)
		if len(parts) > 1 {
			for _, p := range parts {
				p = strings.TrimSpace(p)
				if len(p) > 0 {
					sentences = append(sentences, p)
				}
			}
			return sentences
		}
	}
	return []string{text}
}

func wordSet(text string) map[string]bool {
	words := strings.Fields(text)
	set := make(map[string]bool, len(words))
	for _, w := range words {
		w = strings.Trim(w, ".,!?;:\"'()[]{}")
		if len(w) > 2 { // skip very short words
			set[w] = true
		}
	}
	return set
}

func wordOverlap(a, b map[string]bool) float64 {
	if len(a) == 0 {
		return 0
	}
	overlap := 0
	for w := range a {
		if b[w] {
			overlap++
		}
	}
	return float64(overlap) / float64(len(a))
}
