package server

import (
	"sync"
	"time"
)

// ModelVersion tracks a specific version of a model.
type ModelVersion struct {
	Version   int       `json:"version"`
	Path      string    `json:"path"`
	CreatedAt time.Time `json:"created_at"`
	Active    bool      `json:"active"`
}

// VersionedModelRegistry manages model versions with promote/rollback.
type VersionedModelRegistry struct {
	mu       sync.RWMutex
	models   map[string][]ModelVersion // model_name → versions
	active   map[string]int            // model_name → active version index
}

// NewVersionedModelRegistry creates a versioned registry.
func NewVersionedModelRegistry() *VersionedModelRegistry {
	return &VersionedModelRegistry{
		models: make(map[string][]ModelVersion),
		active: make(map[string]int),
	}
}

// Push adds a new version and makes it active.
func (r *VersionedModelRegistry) Push(name, path string) int {
	r.mu.Lock()
	defer r.mu.Unlock()

	versions := r.models[name]
	// Deactivate all
	for i := range versions {
		versions[i].Active = false
	}

	ver := len(versions) + 1
	versions = append(versions, ModelVersion{
		Version:   ver,
		Path:      path,
		CreatedAt: time.Now(),
		Active:    true,
	})
	r.models[name] = versions
	r.active[name] = ver - 1
	return ver
}

// Rollback reverts to a previous version.
func (r *VersionedModelRegistry) Rollback(name string) (int, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	versions := r.models[name]
	activeIdx := r.active[name]
	if activeIdx <= 0 {
		return 0, false
	}

	versions[activeIdx].Active = false
	activeIdx--
	versions[activeIdx].Active = true
	r.active[name] = activeIdx
	return versions[activeIdx].Version, true
}

// ActiveVersion returns the currently active version.
func (r *VersionedModelRegistry) ActiveVersion(name string) (ModelVersion, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions, ok := r.models[name]
	if !ok {
		return ModelVersion{}, false
	}
	idx := r.active[name]
	return versions[idx], true
}

// ListVersions returns all versions for a model.
func (r *VersionedModelRegistry) ListVersions(name string) []ModelVersion {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions := r.models[name]
	result := make([]ModelVersion, len(versions))
	copy(result, versions)
	return result
}
