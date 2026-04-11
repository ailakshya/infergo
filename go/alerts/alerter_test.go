package alerts

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ailakshya/infergo/analytics"
)

func TestZoneDwellAlert(t *testing.T) {
	rule := ZoneDwellRule("lobby", 5*time.Second)
	am := NewAlertManager([]Rule{rule}, "")

	// Dwell event under threshold: no alert.
	shortDwell := analytics.ZoneEvent{
		Type:      "dwell",
		ZoneName:  "lobby",
		TrackID:   1,
		ClassID:   0,
		Timestamp: time.Now(),
		Duration:  3 * time.Second,
	}
	alerts := am.ProcessEvent(shortDwell, 0, nil)
	if len(alerts) != 0 {
		t.Errorf("expected 0 alerts for short dwell, got %d", len(alerts))
	}

	// Dwell event over threshold: should alert.
	longDwell := analytics.ZoneEvent{
		Type:      "dwell",
		ZoneName:  "lobby",
		TrackID:   1,
		ClassID:   0,
		Timestamp: time.Now(),
		Duration:  7 * time.Second,
	}
	alerts = am.ProcessEvent(longDwell, 0, nil)
	if len(alerts) != 1 {
		t.Errorf("expected 1 alert for long dwell, got %d", len(alerts))
	}
	if len(alerts) > 0 && alerts[0].Rule != rule.Name {
		t.Errorf("alert rule = %q, want %q", alerts[0].Rule, rule.Name)
	}
}

func TestLineCrossCountAlert(t *testing.T) {
	rule := LineCrossCountRule("entrance", 3, 10*time.Second)
	am := NewAlertManager([]Rule{rule}, "")

	crossEvent := analytics.ZoneEvent{
		Type:      "cross",
		ZoneName:  "entrance",
		TrackID:   1,
		ClassID:   0,
		Timestamp: time.Now(),
		Direction: "in",
	}

	// First crossing should trigger alert (condition matches single event).
	alerts := am.ProcessEvent(crossEvent, 0, nil)
	if len(alerts) != 1 {
		t.Errorf("expected 1 alert on cross event, got %d", len(alerts))
	}

	// Subsequent crossings by same track within cooldown should be suppressed.
	crossEvent.TrackID = 1
	alerts = am.ProcessEvent(crossEvent, 0, nil)
	if len(alerts) != 0 {
		t.Errorf("expected 0 alerts within cooldown, got %d", len(alerts))
	}

	// Different track should still trigger.
	crossEvent.TrackID = 2
	alerts = am.ProcessEvent(crossEvent, 0, nil)
	if len(alerts) != 1 {
		t.Errorf("expected 1 alert for different track, got %d", len(alerts))
	}
}

func TestCooldown(t *testing.T) {
	rule := Rule{
		Name: "test_rule",
		Condition: func(event analytics.ZoneEvent) bool {
			return event.Type == "enter"
		},
		Cooldown: 100 * time.Millisecond,
	}
	am := NewAlertManager([]Rule{rule}, "")

	event := analytics.ZoneEvent{
		Type:      "enter",
		ZoneName:  "zone1",
		TrackID:   1,
		ClassID:   0,
		Timestamp: time.Now(),
	}

	// First event should trigger.
	alerts := am.ProcessEvent(event, 0, nil)
	if len(alerts) != 1 {
		t.Fatalf("expected 1 alert, got %d", len(alerts))
	}

	// Immediate second event for same rule+track should be cooled down.
	alerts = am.ProcessEvent(event, 0, nil)
	if len(alerts) != 0 {
		t.Errorf("expected 0 alerts during cooldown, got %d", len(alerts))
	}

	// Wait for cooldown to expire.
	time.Sleep(150 * time.Millisecond)

	// Should trigger again.
	alerts = am.ProcessEvent(event, 0, nil)
	if len(alerts) != 1 {
		t.Errorf("expected 1 alert after cooldown, got %d", len(alerts))
	}
}

func TestWebhookDelivery(t *testing.T) {
	var received atomic.Int32
	var receivedBody []byte

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		received.Add(1)
		buf := make([]byte, r.ContentLength)
		r.Body.Read(buf)
		receivedBody = buf

		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("expected Content-Type application/json, got %s", ct)
		}

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	am := NewAlertManager(nil, server.URL)

	alert := Alert{
		Rule: "test_rule",
		Event: analytics.ZoneEvent{
			Type:     "enter",
			ZoneName: "zone1",
			TrackID:  42,
		},
		StreamID:  1,
		Timestamp: time.Now(),
	}

	err := am.SendWebhook(alert)
	if err != nil {
		t.Fatalf("SendWebhook: %v", err)
	}

	if received.Load() != 1 {
		t.Errorf("expected 1 request, got %d", received.Load())
	}

	// Verify payload structure.
	var parsed Alert
	if err := json.Unmarshal(receivedBody, &parsed); err != nil {
		t.Fatalf("unmarshal webhook body: %v", err)
	}
	if parsed.Rule != "test_rule" {
		t.Errorf("parsed rule = %q, want test_rule", parsed.Rule)
	}
}

func TestWebhookRetry(t *testing.T) {
	var attempts atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := attempts.Add(1)
		if n <= 1 {
			// First attempt fails.
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		// Second attempt succeeds.
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	am := NewAlertManager(nil, server.URL)

	alert := Alert{
		Rule: "retry_test",
		Event: analytics.ZoneEvent{
			Type:     "enter",
			ZoneName: "zone1",
			TrackID:  1,
		},
		StreamID:  1,
		Timestamp: time.Now(),
	}

	err := am.SendWebhook(alert)
	if err != nil {
		t.Fatalf("SendWebhook should succeed on retry: %v", err)
	}

	if attempts.Load() < 2 {
		t.Errorf("expected at least 2 attempts, got %d", attempts.Load())
	}
}

func TestWebhookNoURL(t *testing.T) {
	am := NewAlertManager(nil, "")
	err := am.SendWebhook(Alert{})
	if err == nil {
		t.Error("expected error when no webhook URL configured")
	}
}

func TestZoneOccupancyRule(t *testing.T) {
	rule := ZoneOccupancyRule("entrance", 10)
	if rule.Name == "" {
		t.Error("rule name should not be empty")
	}

	// Enter event for the right zone should match.
	enterEvent := analytics.ZoneEvent{
		Type:     "enter",
		ZoneName: "entrance",
		TrackID:  1,
	}
	if !rule.Condition(enterEvent) {
		t.Error("expected condition to match enter event for correct zone")
	}

	// Exit event should not match.
	exitEvent := analytics.ZoneEvent{
		Type:     "exit",
		ZoneName: "entrance",
		TrackID:  1,
	}
	if rule.Condition(exitEvent) {
		t.Error("expected condition to NOT match exit event")
	}

	// Enter event for wrong zone should not match.
	wrongZone := analytics.ZoneEvent{
		Type:     "enter",
		ZoneName: "other_zone",
		TrackID:  1,
	}
	if rule.Condition(wrongZone) {
		t.Error("expected condition to NOT match enter event for wrong zone")
	}
}

func TestCountEventsInWindow(t *testing.T) {
	am := NewAlertManager(nil, "")

	// Add several events.
	for i := 0; i < 5; i++ {
		event := analytics.ZoneEvent{
			Type:     "cross",
			ZoneName: "line1",
			TrackID:  i + 1,
		}
		am.ProcessEvent(event, 0, nil)
	}

	count := am.CountEventsInWindow("line1", "cross", 10*time.Second)
	if count != 5 {
		t.Errorf("expected 5 events, got %d", count)
	}

	// Count for wrong zone.
	count = am.CountEventsInWindow("line2", "cross", 10*time.Second)
	if count != 0 {
		t.Errorf("expected 0 events for wrong zone, got %d", count)
	}
}

func TestMultipleRules(t *testing.T) {
	rules := []Rule{
		{
			Name:      "rule_a",
			Condition: func(e analytics.ZoneEvent) bool { return e.Type == "enter" },
			Cooldown:  0,
		},
		{
			Name:      "rule_b",
			Condition: func(e analytics.ZoneEvent) bool { return e.Type == "enter" },
			Cooldown:  0,
		},
	}
	am := NewAlertManager(rules, "")

	event := analytics.ZoneEvent{
		Type:     "enter",
		ZoneName: "zone1",
		TrackID:  1,
	}

	alerts := am.ProcessEvent(event, 0, nil)
	if len(alerts) != 2 {
		t.Errorf("expected 2 alerts from 2 matching rules, got %d", len(alerts))
	}

	ruleNames := map[string]bool{}
	for _, a := range alerts {
		ruleNames[a.Rule] = true
	}
	if !ruleNames["rule_a"] || !ruleNames["rule_b"] {
		t.Errorf("expected both rule_a and rule_b, got %v", ruleNames)
	}
}

func TestAlertJSON(t *testing.T) {
	alert := Alert{
		Rule: "test",
		Event: analytics.ZoneEvent{
			Type:      "enter",
			ZoneName:  "zone1",
			TrackID:   1,
			ClassID:   2,
			Timestamp: time.Date(2026, 4, 9, 12, 0, 0, 0, time.UTC),
		},
		StreamID:  1,
		Timestamp: time.Date(2026, 4, 9, 12, 0, 0, 0, time.UTC),
	}

	data, err := json.Marshal(alert)
	if err != nil {
		t.Fatal(err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}

	if parsed["rule"] != "test" {
		t.Errorf("expected rule=test, got %v", parsed["rule"])
	}
	if _, ok := parsed["event"]; !ok {
		t.Error("expected event field in JSON")
	}
	if fmt.Sprintf("%v", parsed["stream_id"]) != "1" {
		t.Errorf("expected stream_id=1, got %v", parsed["stream_id"])
	}
}
