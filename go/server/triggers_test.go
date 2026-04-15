package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestTriggerEngine_Evaluate_Fire(t *testing.T) {
	te := NewTriggerEngine()

	// Set up a test webhook server.
	var received atomic.Int32
	webhookSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		received.Add(1)
		var event TriggerEvent
		json.NewDecoder(r.Body).Decode(&event)
		if event.TriggerName != "person-alert" {
			t.Errorf("expected trigger_name=person-alert, got %q", event.TriggerName)
		}
		if event.MatchCount < 2 {
			t.Errorf("expected at least 2 matches, got %d", event.MatchCount)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer webhookSrv.Close()

	err := te.Add(Trigger{
		Name:                "person-alert",
		ClassID:             0,
		ConfidenceThreshold: 0.5,
		CountThreshold:      2,
		WebhookURL:          webhookSrv.URL,
	})
	if err != nil {
		t.Fatalf("Add error: %v", err)
	}

	detections := []DetectedObject{
		{ClassID: 0, Confidence: 0.9},
		{ClassID: 0, Confidence: 0.8},
		{ClassID: 1, Confidence: 0.7}, // different class, should not match
	}

	fired := te.Evaluate(detections, nil)
	if fired != 1 {
		t.Errorf("expected 1 trigger fired, got %d", fired)
	}

	// Wait for async webhook.
	time.Sleep(100 * time.Millisecond)
	if received.Load() != 1 {
		t.Errorf("expected 1 webhook call, got %d", received.Load())
	}
}

func TestTriggerEngine_Evaluate_Miss(t *testing.T) {
	te := NewTriggerEngine()

	te.Add(Trigger{
		Name:                "person-alert",
		ClassID:             0,
		ConfidenceThreshold: 0.8,
		CountThreshold:      3,
		WebhookURL:          "http://example.com/webhook",
	})

	// Only 2 matching detections, threshold is 3.
	detections := []DetectedObject{
		{ClassID: 0, Confidence: 0.9},
		{ClassID: 0, Confidence: 0.85},
	}

	fired := te.Evaluate(detections, nil)
	if fired != 0 {
		t.Errorf("expected 0 triggers fired (below count threshold), got %d", fired)
	}
}

func TestTriggerEngine_Cooldown(t *testing.T) {
	te := NewTriggerEngine()

	var callCount atomic.Int32
	webhookSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer webhookSrv.Close()

	te.Add(Trigger{
		Name:                "cooldown-test",
		ConfidenceThreshold: 0.5,
		CountThreshold:      1,
		WebhookURL:          webhookSrv.URL,
		Cooldown:            "1h", // very long cooldown
	})

	detections := []DetectedObject{{ClassID: 0, Confidence: 0.9}}

	// First evaluation — should fire.
	fired := te.Evaluate(detections, nil)
	if fired != 1 {
		t.Errorf("first evaluation: expected 1 fired, got %d", fired)
	}

	// Second evaluation — should be suppressed by cooldown.
	fired = te.Evaluate(detections, nil)
	if fired != 0 {
		t.Errorf("second evaluation: expected 0 fired (cooldown), got %d", fired)
	}

	time.Sleep(100 * time.Millisecond)
	if callCount.Load() != 1 {
		t.Errorf("expected exactly 1 webhook call, got %d", callCount.Load())
	}
}

func TestTriggerEngine_ClassName(t *testing.T) {
	te := NewTriggerEngine()

	te.Add(Trigger{
		Name:                "car-alert",
		ClassName:           "car",
		ConfidenceThreshold: 0.5,
		CountThreshold:      1,
		WebhookURL:          "http://example.com/webhook",
	})

	classNames := map[int]string{0: "person", 2: "car"}

	// Detection with class "car" should match.
	detections := []DetectedObject{
		{ClassID: 0, Confidence: 0.9}, // person — no match
		{ClassID: 2, Confidence: 0.8}, // car — match
	}

	fired := te.Evaluate(detections, classNames)
	if fired != 1 {
		t.Errorf("expected 1 trigger fired for class=car, got %d", fired)
	}
}

func TestTriggerEngine_AddRemoveList(t *testing.T) {
	te := NewTriggerEngine()

	te.Add(Trigger{Name: "a", WebhookURL: "http://a.com"})
	te.Add(Trigger{Name: "b", WebhookURL: "http://b.com"})

	list := te.List()
	if len(list) != 2 {
		t.Fatalf("expected 2 triggers, got %d", len(list))
	}

	if !te.Remove("a") {
		t.Error("expected Remove(a) to return true")
	}
	if te.Remove("a") {
		t.Error("expected Remove(a) to return false (already removed)")
	}

	list = te.List()
	if len(list) != 1 {
		t.Fatalf("expected 1 trigger after remove, got %d", len(list))
	}
	if list[0].Name != "b" {
		t.Errorf("expected remaining trigger to be 'b', got %q", list[0].Name)
	}
}

func TestHandleTriggersCreate(t *testing.T) {
	reg := NewRegistry()
	srv := NewServer(reg)
	engine := NewTriggerEngine()
	srv.triggers = engine

	body := `{"name":"person-zone","class_id":0,"confidence_threshold":0.7,"count_threshold":3,"webhook_url":"http://localhost:8080/alert","cooldown":"30s"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/triggers", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	triggers := engine.List()
	if len(triggers) != 1 {
		t.Fatalf("expected 1 trigger, got %d", len(triggers))
	}
	if triggers[0].Name != "person-zone" {
		t.Errorf("expected name=person-zone, got %q", triggers[0].Name)
	}
}

func TestHandleTriggersList(t *testing.T) {
	reg := NewRegistry()
	srv := NewServer(reg)
	engine := NewTriggerEngine()
	srv.triggers = engine

	engine.Add(Trigger{Name: "t1", WebhookURL: "http://a.com"})
	engine.Add(Trigger{Name: "t2", WebhookURL: "http://b.com"})

	req := httptest.NewRequest(http.MethodGet, "/v1/admin/triggers", nil)
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var triggers []Trigger
	json.NewDecoder(rr.Body).Decode(&triggers)
	if len(triggers) != 2 {
		t.Errorf("expected 2 triggers, got %d", len(triggers))
	}
}

func TestHandleTriggersDelete(t *testing.T) {
	reg := NewRegistry()
	srv := NewServer(reg)
	engine := NewTriggerEngine()
	srv.triggers = engine

	engine.Add(Trigger{Name: "deleteme", WebhookURL: "http://a.com"})

	req := httptest.NewRequest(http.MethodDelete, "/v1/admin/triggers/deleteme", nil)
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	if len(engine.List()) != 0 {
		t.Error("expected 0 triggers after delete")
	}
}
