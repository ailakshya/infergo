package server

import (
	"strings"
)

// ContentFilter scans LLM output for harmful content. OPT-93.
type ContentFilter struct {
	mode       string // "block", "warn", "off"
	categories map[string][]string
}

// NewContentFilter creates a content filter.
func NewContentFilter(mode string) *ContentFilter {
	if mode == "" {
		mode = "warn"
	}
	return &ContentFilter{
		mode: mode,
		categories: map[string][]string{
			"violence":  {"kill", "murder", "attack", "weapon", "bomb", "shoot", "stab"},
			"hate":      {"racist", "sexist", "homophobic", "slur", "supremacy"},
			"self_harm": {"suicide", "self-harm", "cut myself", "end my life"},
			"sexual":    {"explicit", "pornographic", "sexually"},
		},
	}
}

// Scan checks text for harmful content. Returns (clean, categories_found).
func (cf *ContentFilter) Scan(text string) (bool, []string) {
	if cf.mode == "off" {
		return true, nil
	}

	lower := strings.ToLower(text)
	var found []string
	for category, keywords := range cf.categories {
		for _, kw := range keywords {
			if strings.Contains(lower, kw) {
				found = append(found, category)
				break
			}
		}
	}

	return len(found) == 0, found
}

// ShouldBlock returns true if content should be blocked (mode=block and harmful).
func (cf *ContentFilter) ShouldBlock(text string) bool {
	if cf.mode != "block" {
		return false
	}
	clean, _ := cf.Scan(text)
	return !clean
}
