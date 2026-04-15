package server

import (
	"os"
	"path/filepath"
	"time"
)

// RetentionPolicy auto-deletes old data files after a configurable period.
type RetentionPolicy struct {
	retentionDays int
	paths         []string // directories to clean
}

// NewRetentionPolicy creates a retention policy.
func NewRetentionPolicy(days int, paths ...string) *RetentionPolicy {
	if days <= 0 {
		days = 30
	}
	return &RetentionPolicy{
		retentionDays: days,
		paths:         paths,
	}
}

// Cleanup removes files older than the retention period.
// Returns number of files deleted.
func (rp *RetentionPolicy) Cleanup() int {
	cutoff := time.Now().Add(-time.Duration(rp.retentionDays) * 24 * time.Hour)
	deleted := 0

	for _, dir := range rp.paths {
		filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() {
				return nil
			}
			if info.ModTime().Before(cutoff) {
				if os.Remove(path) == nil {
					deleted++
				}
			}
			return nil
		})
	}

	return deleted
}

// StartPeriodicCleanup runs cleanup every interval in a goroutine.
func (rp *RetentionPolicy) StartPeriodicCleanup(interval time.Duration, stopCh <-chan struct{}) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				rp.Cleanup()
			case <-stopCh:
				return
			}
		}
	}()
}
