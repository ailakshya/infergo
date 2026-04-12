package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
)

// PromptTemplate is a server-side prompt with variables.
type PromptTemplate struct {
	Name     string `json:"name"`
	Template string `json:"template"` // "Summarize: {{text}}"
	System   string `json:"system,omitempty"`
}

// TemplateStore holds registered prompt templates.
type TemplateStore struct {
	mu        sync.RWMutex
	templates map[string]PromptTemplate
}

// NewTemplateStore creates an empty template store.
func NewTemplateStore() *TemplateStore {
	return &TemplateStore{templates: make(map[string]PromptTemplate)}
}

// Register adds a template.
func (ts *TemplateStore) Register(t PromptTemplate) {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	ts.templates[t.Name] = t
}

// Render fills in variables and returns the expanded prompt.
func (ts *TemplateStore) Render(name string, vars map[string]string) (string, string, bool) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	t, ok := ts.templates[name]
	if !ok {
		return "", "", false
	}
	result := t.Template
	for k, v := range vars {
		result = strings.ReplaceAll(result, "{{"+k+"}}", v)
	}
	return result, t.System, true
}

// List returns all registered templates.
func (ts *TemplateStore) List() []PromptTemplate {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	out := make([]PromptTemplate, 0, len(ts.templates))
	for _, t := range ts.templates {
		out = append(out, t)
	}
	return out
}

func (s *Server) handleTemplates(w http.ResponseWriter, r *http.Request) {
	if s.templates == nil {
		s.templates = NewTemplateStore()
	}
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, s.templates.List())
	case http.MethodPost:
		var t PromptTemplate
		if err := json.NewDecoder(r.Body).Decode(&t); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		if t.Name == "" || t.Template == "" {
			writeError(w, http.StatusBadRequest, "name and template required")
			return
		}
		s.templates.Register(t)
		writeJSON(w, http.StatusOK, map[string]string{"status": "registered", "name": t.Name})
	}
}
