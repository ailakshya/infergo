package server

import (
	"container/list"
	"encoding/binary"
	"hash/fnv"
	"math"
	"sync"
	"time"
)

// ResponseCache is an LRU cache for chat completion responses.
// It stores pre-serialized JSON responses keyed by FNV-1a hash of the
// request parameters (model + messages + temperature + max_tokens + response_format).
// Concurrent-safe via sync.RWMutex.
type ResponseCache struct {
	mu    sync.RWMutex
	cache map[uint64]*list.Element // FNV hash → list element
	order *list.List               // LRU eviction order (front = most recent)
	size  int                      // max entries
}

// cacheEntry holds a cached response and metadata.
type cacheEntry struct {
	key      uint64
	response []byte    // pre-serialized JSON response
	created  time.Time
	hits     int
}

// NewResponseCache creates a response cache with the given max entry count.
// If size <= 0, caching is effectively disabled (Get always misses).
func NewResponseCache(size int) *ResponseCache {
	return &ResponseCache{
		cache: make(map[uint64]*list.Element, size),
		order: list.New(),
		size:  size,
	}
}

// Get looks up a cached response by request key. Returns the pre-serialized
// JSON bytes and true on hit, or nil and false on miss.
func (c *ResponseCache) Get(key uint64) ([]byte, bool) {
	if c == nil || c.size <= 0 {
		return nil, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	elem, ok := c.cache[key]
	if !ok {
		return nil, false
	}
	entry := elem.Value.(*cacheEntry)
	entry.hits++
	// Move to front (most recently used).
	c.order.MoveToFront(elem)
	return entry.response, true
}

// Put stores a pre-serialized response under the given key. If the cache
// is full, the least recently used entry is evicted.
func (c *ResponseCache) Put(key uint64, response []byte) {
	if c == nil || c.size <= 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	// Update existing entry if present.
	if elem, ok := c.cache[key]; ok {
		entry := elem.Value.(*cacheEntry)
		entry.response = response
		entry.created = time.Now()
		c.order.MoveToFront(elem)
		return
	}

	// Evict LRU entry if at capacity.
	if c.order.Len() >= c.size {
		tail := c.order.Back()
		if tail != nil {
			evicted := c.order.Remove(tail).(*cacheEntry)
			delete(c.cache, evicted.key)
		}
	}

	// Insert new entry at front.
	entry := &cacheEntry{
		key:      key,
		response: response,
		created:  time.Now(),
	}
	elem := c.order.PushFront(entry)
	c.cache[key] = elem
}

// Len returns the current number of entries in the cache.
func (c *ResponseCache) Len() int {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.order.Len()
}

// CacheKey computes an FNV-1a hash from the request parameters that affect
// the response: model name, message contents, temperature, max_tokens,
// and response_format type. Different parameters produce different keys;
// identical parameters produce the same key.
func CacheKey(req *ChatCompletionRequest) uint64 {
	h := fnv.New64a()

	// Model
	h.Write([]byte(req.Model))
	h.Write([]byte{0}) // separator

	// Messages: role + content for each
	for _, m := range req.Messages {
		h.Write([]byte(m.Role))
		h.Write([]byte{0})
		h.Write([]byte(m.Content))
		h.Write([]byte{0})
	}

	// Temperature: encode as 8 bytes
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], math.Float64bits(float64(req.Temp)))
	h.Write(buf[:])

	// MaxTokens
	binary.LittleEndian.PutUint64(buf[:], uint64(req.MaxTokens))
	h.Write(buf[:])

	// ResponseFormat type (if set)
	if req.ResponseFormat != nil {
		h.Write([]byte(req.ResponseFormat.Type))
	}
	h.Write([]byte{0})

	return h.Sum64()
}
