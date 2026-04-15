package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// DataConnector represents a configured external data source. OPT-74.
type DataConnector struct {
	ID       string            `json:"id"`
	Type     string            `json:"type"`     // "postgres", "mongodb", "s3", "http"
	Config   map[string]string `json:"config"`   // connection params
	SyncSec  int               `json:"sync_sec"` // auto-sync interval (0 = manual)
	LastSync time.Time         `json:"last_sync"`
	Status   string            `json:"status"`   // "active", "error", "paused"
}

// ConnectorStore manages data connectors.
type ConnectorStore struct {
	mu         sync.RWMutex
	connectors map[string]*DataConnector
}

// NewConnectorStore creates a connector store.
func NewConnectorStore() *ConnectorStore {
	return &ConnectorStore{
		connectors: make(map[string]*DataConnector),
	}
}

// Add registers a new connector.
func (cs *ConnectorStore) Add(c DataConnector) {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	c.Status = "active"
	cs.connectors[c.ID] = &c
}

// Get returns a connector by ID.
func (cs *ConnectorStore) Get(id string) (*DataConnector, bool) {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	c, ok := cs.connectors[id]
	return c, ok
}

// List returns all connectors.
func (cs *ConnectorStore) List() []DataConnector {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	result := make([]DataConnector, 0, len(cs.connectors))
	for _, c := range cs.connectors {
		result = append(result, *c)
	}
	return result
}

// Delete removes a connector.
func (cs *ConnectorStore) Delete(id string) bool {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	_, ok := cs.connectors[id]
	delete(cs.connectors, id)
	return ok
}

func (s *Server) handleConnectors(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET":
		store := NewConnectorStore() // placeholder — should be on Server struct
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"connectors": store.List(),
		})
	case "POST":
		var c DataConnector
		if err := json.NewDecoder(r.Body).Decode(&c); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		if c.ID == "" {
			c.ID = fmt.Sprintf("conn-%d", time.Now().UnixNano())
		}
		writeJSON(w, http.StatusCreated, map[string]interface{}{
			"status": "created", "id": c.ID,
		})
	}
}
