package server

import (
	"sync"
	"time"
)

// PromptVersion tracks versions of prompt templates.
type PromptVersion struct {
	Version   int       `json:"version"`
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"created_at"`
}

// PromptLibrary stores named prompt templates with versioning.
type PromptLibrary struct {
	mu       sync.RWMutex
	prompts  map[string][]PromptVersion // name → versions (index 0 = oldest)
}

// NewPromptLibrary creates a prompt library.
func NewPromptLibrary() *PromptLibrary {
	return &PromptLibrary{
		prompts: make(map[string][]PromptVersion),
	}
}

// Set creates or updates a prompt (creates new version).
func (pl *PromptLibrary) Set(name, content string) int {
	pl.mu.Lock()
	defer pl.mu.Unlock()

	versions := pl.prompts[name]
	ver := len(versions) + 1
	pl.prompts[name] = append(versions, PromptVersion{
		Version:   ver,
		Content:   content,
		CreatedAt: time.Now(),
	})
	return ver
}

// Get returns the latest version of a prompt.
func (pl *PromptLibrary) Get(name string) (PromptVersion, bool) {
	pl.mu.RLock()
	defer pl.mu.RUnlock()

	versions, ok := pl.prompts[name]
	if !ok || len(versions) == 0 {
		return PromptVersion{}, false
	}
	return versions[len(versions)-1], true
}

// GetVersion returns a specific version.
func (pl *PromptLibrary) GetVersion(name string, ver int) (PromptVersion, bool) {
	pl.mu.RLock()
	defer pl.mu.RUnlock()

	versions, ok := pl.prompts[name]
	if !ok || ver < 1 || ver > len(versions) {
		return PromptVersion{}, false
	}
	return versions[ver-1], true
}

// Rollback sets the active version to a previous one (creates a new version with old content).
func (pl *PromptLibrary) Rollback(name string, ver int) (int, bool) {
	pl.mu.Lock()
	defer pl.mu.Unlock()

	versions, ok := pl.prompts[name]
	if !ok || ver < 1 || ver > len(versions) {
		return 0, false
	}

	old := versions[ver-1]
	newVer := len(versions) + 1
	pl.prompts[name] = append(versions, PromptVersion{
		Version:   newVer,
		Content:   old.Content,
		CreatedAt: time.Now(),
	})
	return newVer, true
}

// List returns all prompt names.
func (pl *PromptLibrary) List() []string {
	pl.mu.RLock()
	defer pl.mu.RUnlock()

	names := make([]string, 0, len(pl.prompts))
	for name := range pl.prompts {
		names = append(names, name)
	}
	return names
}

// Versions returns all versions for a prompt.
func (pl *PromptLibrary) Versions(name string) []PromptVersion {
	pl.mu.RLock()
	defer pl.mu.RUnlock()

	versions, ok := pl.prompts[name]
	if !ok {
		return nil
	}
	result := make([]PromptVersion, len(versions))
	copy(result, versions)
	return result
}
