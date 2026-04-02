package server_test

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ailakshya/infergo/server"
)

// ─── mock model ──────────────────────────────────────────────────────────────

type mockModel struct {
	id      int
	closed  atomic.Bool
	onClose func(id int)
}

func (m *mockModel) Close() {
	m.closed.Store(true)
	if m.onClose != nil {
		m.onClose(m.id)
	}
}

var modelIDSeq atomic.Int32

func newMock(onClose func(id int)) *mockModel {
	return &mockModel{id: int(modelIDSeq.Add(1)), onClose: onClose}
}

// ─── Construction / basic ops ────────────────────────────────────────────────

func TestRegistry_LoadAndGet(t *testing.T) {
	r := server.NewRegistry()
	m := newMock(nil)
	if err := r.Load("foo", m); err != nil {
		t.Fatalf("Load: %v", err)
	}

	ref, err := r.Get("foo")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if ref.Model != m {
		t.Error("expected same model back")
	}
	ref.Release()
}

func TestRegistry_GetMissing(t *testing.T) {
	r := server.NewRegistry()
	if _, err := r.Get("nope"); err == nil {
		t.Error("expected error for missing model")
	}
}

func TestRegistry_LoadEmptyName(t *testing.T) {
	r := server.NewRegistry()
	if err := r.Load("", newMock(nil)); err == nil {
		t.Error("expected error for empty name")
	}
}

func TestRegistry_LoadNilModel(t *testing.T) {
	r := server.NewRegistry()
	if err := r.Load("foo", nil); err == nil {
		t.Error("expected error for nil model")
	}
}

func TestRegistry_Unload(t *testing.T) {
	r := server.NewRegistry()
	closed := make(chan int, 1)
	m := newMock(func(id int) { closed <- id })
	if err := r.Load("foo", m); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := r.Unload("foo"); err != nil {
		t.Fatalf("Unload: %v", err)
	}
	// Model with no refs → closed immediately
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Error("model not closed after Unload")
	}
	if _, err := r.Get("foo"); err == nil {
		t.Error("expected error after Unload")
	}
}

func TestRegistry_UnloadMissing(t *testing.T) {
	r := server.NewRegistry()
	if err := r.Unload("nope"); err == nil {
		t.Error("expected error unloading missing model")
	}
}

func TestRegistry_Names(t *testing.T) {
	r := server.NewRegistry()
	r.Load("a", newMock(nil))
	r.Load("b", newMock(nil))
	names := r.Names()
	if len(names) != 2 {
		t.Errorf("expected 2 names, got %v", names)
	}
}

// ─── Hot reload ───────────────────────────────────────────────────────────────

func TestRegistry_HotReload_InFlightRequestCompletes(t *testing.T) {
	r := server.NewRegistry()
	closedIDs := make(chan int, 2)
	onClose := func(id int) { closedIDs <- id }

	m1 := newMock(onClose)
	m2 := newMock(onClose)

	r.Load("svc", m1)

	// Acquire a ref to m1 (simulating an in-flight request)
	ref, err := r.Get("svc")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if ref.Model != m1 {
		t.Error("expected m1")
	}

	// Hot reload: m2 replaces m1
	r.Load("svc", m2)

	// m1 should NOT be closed yet (ref is still held)
	select {
	case id := <-closedIDs:
		t.Fatalf("model %d closed while ref still held", id)
	case <-time.After(20 * time.Millisecond):
	}

	// New Get sees m2
	ref2, err := r.Get("svc")
	if err != nil {
		t.Fatalf("Get after reload: %v", err)
	}
	if ref2.Model != m2 {
		t.Error("expected m2 after reload")
	}
	ref2.Release()

	// Release the m1 ref → m1 now closes
	ref.Release()
	select {
	case id := <-closedIDs:
		if id != m1.id {
			t.Errorf("expected m1 (id=%d) to close, got id=%d", m1.id, id)
		}
	case <-time.After(time.Second):
		t.Error("m1 not closed after Release")
	}
}

// ─── 100 concurrent requests + hot reload ────────────────────────────────────

func TestRegistry_100ConcurrentRequests_NoDropped(t *testing.T) {
	r := server.NewRegistry()

	var served atomic.Int64
	var wg sync.WaitGroup
	const N = 100

	r.Load("model", newMock(nil))

	// Launch 100 goroutines that each Get + work + Release
	wg.Add(N)
	for i := range N {
		go func(i int) {
			defer wg.Done()
			// Retry in case we catch a brief reload window
			for retry := range 10 {
				ref, err := r.Get("model")
				if err != nil {
					if retry == 9 {
						t.Errorf("request %d: Get failed after retries: %v", i, err)
					}
					time.Sleep(time.Millisecond)
					continue
				}
				// Simulate work
				time.Sleep(time.Microsecond)
				ref.Release()
				served.Add(1)
				return
			}
		}(i)
	}

	// Concurrently hot-reload several times while requests fly
	go func() {
		for range 5 {
			time.Sleep(2 * time.Millisecond)
			r.Load("model", newMock(nil))
		}
	}()

	wg.Wait()

	if got := served.Load(); got != N {
		t.Errorf("expected %d served, got %d", N, got)
	}
}

// ─── Multiple model types coexist ────────────────────────────────────────────

func TestRegistry_MultipleModels(t *testing.T) {
	r := server.NewRegistry()
	for i := range 5 {
		r.Load(fmt.Sprintf("model-%d", i), newMock(nil))
	}
	if len(r.Names()) != 5 {
		t.Errorf("expected 5 models, got %d", len(r.Names()))
	}
	for i := range 5 {
		ref, err := r.Get(fmt.Sprintf("model-%d", i))
		if err != nil {
			t.Errorf("Get model-%d: %v", i, err)
			continue
		}
		ref.Release()
	}
}
