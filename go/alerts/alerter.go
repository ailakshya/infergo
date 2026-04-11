// Package alerts provides an event-driven alerting system that processes
// zone analytics events and triggers configurable actions such as webhook
// notifications.
package alerts

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/ailakshya/infergo/analytics"
)

// ConditionFunc tests whether a zone event should trigger an alert.
type ConditionFunc func(event analytics.ZoneEvent) bool

// ActionFunc executes an action when an alert is triggered.
type ActionFunc func(alert Alert) error

// Rule defines an alerting rule with a condition, action, and cooldown.
type Rule struct {
	Name      string        `json:"name"`
	Condition ConditionFunc `json:"-"`
	Action    ActionFunc    `json:"-"`
	Cooldown  time.Duration `json:"cooldown"`
}

// Alert represents a triggered alert.
type Alert struct {
	Rule      string              `json:"rule"`
	Event     analytics.ZoneEvent `json:"event"`
	StreamID  int                 `json:"stream_id"`
	Timestamp time.Time           `json:"timestamp"`
	Snapshot  []byte              `json:"-"` // JPEG snapshot (excluded from JSON for size)
}

// AlertManager processes zone events against rules and dispatches alerts.
type AlertManager struct {
	rules   []Rule
	webhook string
	client  *http.Client

	mu        sync.Mutex
	cooldowns map[string]time.Time // "ruleName:trackID" -> last alert time

	// For count-based rules: track event counts within windows.
	eventLog   []timedEvent
	eventLogMu sync.Mutex
}

// timedEvent stores a zone event with its receipt time, used for windowed counting.
type timedEvent struct {
	event analytics.ZoneEvent
	at    time.Time
}

// NewAlertManager creates a new AlertManager with the given rules and optional webhook URL.
func NewAlertManager(rules []Rule, webhookURL string) *AlertManager {
	return &AlertManager{
		rules:     rules,
		webhook:   webhookURL,
		cooldowns: make(map[string]time.Time),
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// ProcessEvent evaluates all rules against the given event and returns triggered alerts.
// It enforces cooldown periods per rule+track combination.
func (am *AlertManager) ProcessEvent(event analytics.ZoneEvent, streamID int, snapshot []byte) []Alert {
	am.mu.Lock()
	defer am.mu.Unlock()

	// Record event for windowed counting.
	am.eventLogMu.Lock()
	am.eventLog = append(am.eventLog, timedEvent{event: event, at: time.Now()})
	am.eventLogMu.Unlock()

	var alerts []Alert

	for _, rule := range am.rules {
		if rule.Condition == nil || !rule.Condition(event) {
			continue
		}

		// Check cooldown.
		key := fmt.Sprintf("%s:%d", rule.Name, event.TrackID)
		if lastTime, ok := am.cooldowns[key]; ok {
			if time.Since(lastTime) < rule.Cooldown {
				continue
			}
		}

		alert := Alert{
			Rule:      rule.Name,
			Event:     event,
			StreamID:  streamID,
			Timestamp: time.Now(),
			Snapshot:  snapshot,
		}

		// Execute action if defined.
		if rule.Action != nil {
			if err := rule.Action(alert); err != nil {
				// Log but don't fail: alerting is best-effort.
				continue
			}
		}

		am.cooldowns[key] = time.Now()
		alerts = append(alerts, alert)
	}

	return alerts
}

// SendWebhook sends an alert as a JSON POST to the configured webhook URL.
// It retries up to 3 times with exponential backoff on failure.
func (am *AlertManager) SendWebhook(alert Alert) error {
	if am.webhook == "" {
		return fmt.Errorf("no webhook URL configured")
	}

	payload, err := json.Marshal(alert)
	if err != nil {
		return fmt.Errorf("marshal alert: %w", err)
	}

	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(math.Pow(2, float64(attempt))) * 100 * time.Millisecond
			time.Sleep(backoff)
		}

		req, err := http.NewRequest(http.MethodPost, am.webhook, bytes.NewReader(payload))
		if err != nil {
			return fmt.Errorf("create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := am.client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		resp.Body.Close()

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return nil
		}
		lastErr = fmt.Errorf("webhook returned status %d", resp.StatusCode)
	}

	return fmt.Errorf("webhook failed after 3 attempts: %w", lastErr)
}

// CountEventsInWindow counts events matching the given zone name and event type
// within the specified time window.
func (am *AlertManager) CountEventsInWindow(zoneName, eventType string, window time.Duration) int {
	am.eventLogMu.Lock()
	defer am.eventLogMu.Unlock()

	cutoff := time.Now().Add(-window)
	count := 0

	// Prune old events while counting.
	pruneIdx := 0
	for i, te := range am.eventLog {
		if te.at.Before(cutoff) {
			pruneIdx = i + 1
			continue
		}
		if te.event.ZoneName == zoneName && te.event.Type == eventType {
			count++
		}
	}

	// Prune old entries.
	if pruneIdx > 0 {
		am.eventLog = am.eventLog[pruneIdx:]
	}

	return count
}

// --- Built-in rule constructors ---

// ZoneDwellRule creates a rule that fires when a track has been dwelling
// in the named zone longer than the given threshold.
func ZoneDwellRule(zoneName string, threshold time.Duration) Rule {
	return Rule{
		Name: fmt.Sprintf("dwell_%s_%s", zoneName, threshold),
		Condition: func(event analytics.ZoneEvent) bool {
			return event.Type == "dwell" &&
				event.ZoneName == zoneName &&
				event.Duration >= threshold
		},
		Cooldown: threshold, // Don't re-alert for the same dwell within the threshold period.
	}
}

// LineCrossCountRule creates a rule that fires when a line has been crossed
// at least `count` times within the given time window. The rule is checked on
// each "cross" event for the named line.
func LineCrossCountRule(lineName string, count int, window time.Duration) Rule {
	return Rule{
		Name: fmt.Sprintf("line_count_%s_%d_%s", lineName, count, window),
		Condition: func(event analytics.ZoneEvent) bool {
			// This is a placeholder condition that matches cross events for the line.
			// The actual count check should be done by the caller using CountEventsInWindow.
			return event.Type == "cross" && event.ZoneName == lineName
		},
		Cooldown: window,
	}
}

// ZoneOccupancyRule creates a rule that fires on zone enter events.
// The caller should check the current occupancy count via ZoneManager.Occupancy()
// and only process the alert if occupancy exceeds maxOccupancy.
func ZoneOccupancyRule(zoneName string, maxOccupancy int) Rule {
	return Rule{
		Name: fmt.Sprintf("occupancy_%s_%d", zoneName, maxOccupancy),
		Condition: func(event analytics.ZoneEvent) bool {
			return event.Type == "enter" && event.ZoneName == zoneName
		},
		Cooldown: 5 * time.Second,
	}
}
