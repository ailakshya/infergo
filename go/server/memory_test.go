package server

import (
	"fmt"
	"testing"
)

func TestMemoryAppendAndGet(t *testing.T) {
	m := NewConversationMemory(10, 50)

	msgs := []ChatMessage{{Role: "user", Content: "Hello"}}
	result := m.Append("s1", msgs)

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].Content != "Hello" {
		t.Fatalf("expected 'Hello', got %q", result[0].Content)
	}

	// Append more
	result = m.Append("s1", []ChatMessage{{Role: "assistant", Content: "Hi there"}})
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}

	// Get
	got := m.Get("s1")
	if len(got) != 2 {
		t.Fatalf("Get: expected 2, got %d", len(got))
	}
}

func TestMemorySessionIsolation(t *testing.T) {
	m := NewConversationMemory(10, 50)

	m.Append("s1", []ChatMessage{{Role: "user", Content: "Session 1"}})
	m.Append("s2", []ChatMessage{{Role: "user", Content: "Session 2"}})

	s1 := m.Get("s1")
	s2 := m.Get("s2")

	if len(s1) != 1 || s1[0].Content != "Session 1" {
		t.Fatalf("s1 wrong: %v", s1)
	}
	if len(s2) != 1 || s2[0].Content != "Session 2" {
		t.Fatalf("s2 wrong: %v", s2)
	}
}

func TestMemorySlidingWindow(t *testing.T) {
	m := NewConversationMemory(10, 5) // max 5 messages

	// Add system + 6 user messages
	m.Append("s1", []ChatMessage{{Role: "system", Content: "You are helpful"}})
	for i := 0; i < 6; i++ {
		m.Append("s1", []ChatMessage{{Role: "user", Content: fmt.Sprintf("msg%d", i)}})
	}

	got := m.Get("s1")
	if len(got) > 5 {
		t.Fatalf("expected ≤5 messages after sliding window, got %d", len(got))
	}
	// System message should be preserved
	if got[0].Role != "system" {
		t.Fatalf("system message should be preserved, got role=%s", got[0].Role)
	}
}

func TestMemoryLRUEviction(t *testing.T) {
	m := NewConversationMemory(3, 50) // max 3 sessions

	m.Append("s1", []ChatMessage{{Role: "user", Content: "1"}})
	m.Append("s2", []ChatMessage{{Role: "user", Content: "2"}})
	m.Append("s3", []ChatMessage{{Role: "user", Content: "3"}})
	m.Append("s4", []ChatMessage{{Role: "user", Content: "4"}}) // s1 evicted

	if m.Get("s1") != nil {
		t.Fatal("s1 should have been evicted")
	}
	if m.Get("s4") == nil {
		t.Fatal("s4 should exist")
	}
	if m.Size() != 3 {
		t.Fatalf("expected 3 sessions, got %d", m.Size())
	}
}

func TestMemoryDelete(t *testing.T) {
	m := NewConversationMemory(10, 50)

	m.Append("s1", []ChatMessage{{Role: "user", Content: "Hello"}})
	if !m.Delete("s1") {
		t.Fatal("delete should return true")
	}
	if m.Get("s1") != nil {
		t.Fatal("s1 should be gone after delete")
	}
	if m.Delete("s1") {
		t.Fatal("double delete should return false")
	}
}

func TestMemoryNoSession(t *testing.T) {
	m := NewConversationMemory(10, 50)
	if m.Get("nonexistent") != nil {
		t.Fatal("nonexistent session should return nil")
	}
}
