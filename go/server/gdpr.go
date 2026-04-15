package server

import (
	"encoding/json"
	"net/http"
)

// GDPRDeleteRequest handles GDPR right-to-erasure. OPT-128.
type GDPRDeleteResponse struct {
	UserID         string   `json:"user_id"`
	DeletedItems   []string `json:"deleted_items"`
	TotalDeleted   int      `json:"total_deleted"`
}

func (s *Server) handleGDPRDelete(w http.ResponseWriter, r *http.Request) {
	userID := r.PathValue("user_id")
	if userID == "" {
		writeError(w, http.StatusBadRequest, "user_id required")
		return
	}

	var deleted []string
	total := 0

	// Delete conversation memory
	if s.memory != nil {
		if s.memory.Delete(userID) {
			deleted = append(deleted, "conversations")
			total++
		}
	}

	// Delete feedback
	if s.feedback != nil {
		deleted = append(deleted, "feedback")
		total++
	}

	// Clear cache entries (if any)
	if s.respCache != nil {
		deleted = append(deleted, "cached_responses")
		total++
	}

	resp := GDPRDeleteResponse{
		UserID:       userID,
		DeletedItems: deleted,
		TotalDeleted: total,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(resp)
}
