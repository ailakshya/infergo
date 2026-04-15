package server

import (
	"strings"
)

// MoERouter routes queries to specialized models based on topic. OPT-121.
type MoERouter struct {
	routes   map[string][]string // keyword patterns → model name
	fallback string             // default model
}

// NewMoERouter creates a mixture-of-experts router.
func NewMoERouter(fallback string) *MoERouter {
	return &MoERouter{
		routes:   make(map[string][]string),
		fallback: fallback,
	}
}

// AddRoute adds keyword patterns that route to a specific model.
func (m *MoERouter) AddRoute(model string, keywords []string) {
	m.routes[model] = keywords
}

// Route returns the best model for the query.
func (m *MoERouter) Route(query string) string {
	lower := strings.ToLower(query)

	bestModel := m.fallback
	bestScore := 0

	for model, keywords := range m.routes {
		score := 0
		for _, kw := range keywords {
			if strings.Contains(lower, strings.ToLower(kw)) {
				score++
			}
		}
		if score > bestScore {
			bestScore = score
			bestModel = model
		}
	}

	return bestModel
}

// DefaultCodeRouter creates a router for code/math/general queries.
func DefaultCodeRouter(codeModel, mathModel, generalModel string) *MoERouter {
	r := NewMoERouter(generalModel)
	r.AddRoute(codeModel, []string{
		"code", "function", "class", "import", "def ", "var ", "const ",
		"python", "javascript", "golang", "rust", "java", "typescript",
		"debug", "compile", "syntax", "algorithm", "api", "json",
		"html", "css", "sql", "database", "query", "regex",
	})
	r.AddRoute(mathModel, []string{
		"calculate", "equation", "solve", "integral", "derivative",
		"probability", "statistics", "matrix", "algebra", "geometry",
		"theorem", "proof", "formula", "math", "number",
	})
	return r
}
