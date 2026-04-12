package server

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"
)

// SemanticCache caches LLM responses by query hash.
// Similar queries hit the cache and return instantly (0ms).
type SemanticCache struct {
	mu      sync.RWMutex
	entries map[string]*cacheEntry
	maxSize int
	ttl     time.Duration
}

type cacheEntry struct {
	response  string
	createdAt time.Time
}

// NewSemanticCache creates a cache with max entries and TTL.
func NewSemanticCache(maxSize int, ttl time.Duration) *SemanticCache {
	return &SemanticCache{
		entries: make(map[string]*cacheEntry),
		maxSize: maxSize,
		ttl:     ttl,
	}
}

// Get returns cached response if exists and not expired.
func (c *SemanticCache) Get(query string) (string, bool) {
	key := hashQuery(query)
	c.mu.RLock()
	defer c.mu.RUnlock()
	e, ok := c.entries[key]
	if !ok || time.Since(e.createdAt) > c.ttl {
		return "", false
	}
	return e.response, true
}

// Put stores a response in the cache.
func (c *SemanticCache) Put(query, response string) {
	key := hashQuery(query)
	c.mu.Lock()
	defer c.mu.Unlock()
	// Evict oldest if full
	if len(c.entries) >= c.maxSize {
		var oldest string
		var oldestTime time.Time
		for k, v := range c.entries {
			if oldest == "" || v.createdAt.Before(oldestTime) {
				oldest = k
				oldestTime = v.createdAt
			}
		}
		delete(c.entries, oldest)
	}
	c.entries[key] = &cacheEntry{response: response, createdAt: time.Now()}
}

// Size returns number of cached entries.
func (c *SemanticCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

func hashQuery(q string) string {
	h := sha256.Sum256([]byte(q))
	return hex.EncodeToString(h[:16])
}
