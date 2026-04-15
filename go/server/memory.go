package server

import (
	"container/list"
	"sync"
	"time"
)

// ConversationMemory stores multi-turn chat history per session.
// LRU eviction when max sessions exceeded. Thread-safe.
type ConversationMemory struct {
	mu       sync.RWMutex
	sessions map[string]*sessionEntry
	order    *list.List
	maxSess  int
	maxMsgs  int // max messages per session before sliding window
}

type sessionEntry struct {
	id       string
	messages []ChatMessage
	created  time.Time
	updated  time.Time
	element  *list.Element
}

// NewConversationMemory creates a memory store.
// maxSessions: max concurrent sessions (LRU eviction beyond this).
// maxMessages: max messages per session (oldest dropped when exceeded).
func NewConversationMemory(maxSessions, maxMessages int) *ConversationMemory {
	if maxSessions <= 0 {
		maxSessions = 1000
	}
	if maxMessages <= 0 {
		maxMessages = 100
	}
	return &ConversationMemory{
		sessions: make(map[string]*sessionEntry),
		order:    list.New(),
		maxSess:  maxSessions,
		maxMsgs:  maxMessages,
	}
}

// Append adds new messages to a session and returns the full history.
// Creates the session if it doesn't exist.
func (m *ConversationMemory) Append(sessionID string, msgs []ChatMessage) []ChatMessage {
	m.mu.Lock()
	defer m.mu.Unlock()

	entry, ok := m.sessions[sessionID]
	if !ok {
		// New session
		if m.order.Len() >= m.maxSess {
			// Evict oldest
			oldest := m.order.Back()
			if oldest != nil {
				old := oldest.Value.(*sessionEntry)
				delete(m.sessions, old.id)
				m.order.Remove(oldest)
			}
		}
		entry = &sessionEntry{
			id:       sessionID,
			messages: make([]ChatMessage, 0, 16),
			created:  time.Now(),
		}
		entry.element = m.order.PushFront(entry)
		m.sessions[sessionID] = entry
	} else {
		// Move to front (most recently used)
		m.order.MoveToFront(entry.element)
	}

	entry.messages = append(entry.messages, msgs...)
	entry.updated = time.Now()

	// Sliding window: keep system message + last N messages
	if len(entry.messages) > m.maxMsgs {
		// Preserve first message if it's a system message
		start := 0
		if len(entry.messages) > 0 && entry.messages[0].Role == "system" {
			start = 1
		}
		excess := len(entry.messages) - m.maxMsgs
		if excess > 0 && start < len(entry.messages) {
			// Remove oldest non-system messages
			entry.messages = append(entry.messages[:start], entry.messages[start+excess:]...)
		}
	}

	// Return a copy
	result := make([]ChatMessage, len(entry.messages))
	copy(result, entry.messages)
	return result
}

// Get returns the current history for a session. Returns nil if not found.
func (m *ConversationMemory) Get(sessionID string) []ChatMessage {
	m.mu.RLock()
	defer m.mu.RUnlock()

	entry, ok := m.sessions[sessionID]
	if !ok {
		return nil
	}
	result := make([]ChatMessage, len(entry.messages))
	copy(result, entry.messages)
	return result
}

// Delete removes a session.
func (m *ConversationMemory) Delete(sessionID string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	entry, ok := m.sessions[sessionID]
	if !ok {
		return false
	}
	m.order.Remove(entry.element)
	delete(m.sessions, sessionID)
	return true
}

// Size returns the number of active sessions.
func (m *ConversationMemory) Size() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.sessions)
}
