package server

import (
	"errors"
	"fmt"
	"sync"
)

// Model is the interface that all registered models must implement.
// Close releases all C-side resources; it is called automatically when the
// last in-flight request completes after the model has been unloaded.
type Model interface {
	Close()
}

// ModelRef is a short-lived handle returned by Registry.Get.
// Call Release when the model is no longer needed; do not hold it longer
// than a single request.
type ModelRef struct {
	// Model is the underlying model; safe to use until Release is called.
	Model Model
	entry *modelEntry
}

// Release decrements the reference count. If the model has been unloaded
// and this was the last reference, Close is called.
func (r *ModelRef) Release() {
	r.entry.decref()
}

// modelEntry is the internal per-model record.
type modelEntry struct {
	mu      sync.Mutex
	model   Model
	refs    int
	closing bool
	closed  chan struct{} // closed when refs reach 0 after closing=true
}

func newEntry(m Model) *modelEntry {
	return &modelEntry{
		model:  m,
		closed: make(chan struct{}),
	}
}

// incref increments the reference count.
// Returns false if the entry is already closing (caller must not use it).
func (e *modelEntry) incref() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closing {
		return false
	}
	e.refs++
	return true
}

// decref decrements the reference count.
// If refs reach 0 and closing is set, the model is closed.
func (e *modelEntry) decref() {
	e.mu.Lock()
	e.refs--
	shouldClose := e.closing && e.refs == 0
	e.mu.Unlock()
	if shouldClose {
		e.model.Close()
		close(e.closed)
	}
}

// scheduleClose marks the entry for closing. If there are no current
// references the model is closed immediately; otherwise it is deferred.
func (e *modelEntry) scheduleClose() {
	e.mu.Lock()
	e.closing = true
	shouldClose := e.refs == 0
	e.mu.Unlock()
	if shouldClose {
		e.model.Close()
		close(e.closed)
	}
}

// Registry maps names to models and supports hot reload.
// It is safe for concurrent use.
type Registry struct {
	mu      sync.RWMutex
	entries map[string]*modelEntry
}

// NewRegistry returns an empty Registry.
func NewRegistry() *Registry {
	return &Registry{entries: make(map[string]*modelEntry)}
}

// Load registers model under name. If a model with that name is already
// registered it is replaced atomically — in-flight requests continue using
// the old model until they call Release; new Get calls see the new model.
// Load takes ownership of m; m.Close() will be called when the entry is
// eventually evicted.
func (r *Registry) Load(name string, m Model) error {
	if name == "" {
		return errors.New("registry: name must not be empty")
	}
	if m == nil {
		return errors.New("registry: model must not be nil")
	}

	r.mu.Lock()
	old := r.entries[name]
	r.entries[name] = newEntry(m)
	r.mu.Unlock()

	if old != nil {
		old.scheduleClose()
	}
	return nil
}

// Unload removes the named model. In-flight requests continue to completion;
// m.Close() is called once the last reference is released.
// Returns an error if the model is not found.
func (r *Registry) Unload(name string) error {
	r.mu.Lock()
	e, ok := r.entries[name]
	if ok {
		delete(r.entries, name)
	}
	r.mu.Unlock()

	if !ok {
		return fmt.Errorf("registry: model %q not found", name)
	}
	e.scheduleClose()
	return nil
}

// Get returns a ModelRef for the named model.
// The caller must call ModelRef.Release() when done.
// Returns an error if the model is not found.
func (r *Registry) Get(name string) (*ModelRef, error) {
	r.mu.RLock()
	e, ok := r.entries[name]
	r.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("registry: model %q not found", name)
	}
	if !e.incref() {
		return nil, fmt.Errorf("registry: model %q is being unloaded", name)
	}
	return &ModelRef{Model: e.model, entry: e}, nil
}

// Names returns the names of all currently registered models.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.entries))
	for k := range r.entries {
		names = append(names, k)
	}
	return names
}

// WaitUnloaded blocks until the named model (previously obtained via Unload
// or a superseded Load) has been fully closed. Useful in tests.
// Pass the entry obtained before Unload to avoid a TOCTOU race.
func waitEntryDone(e *modelEntry) {
	<-e.closed
}
