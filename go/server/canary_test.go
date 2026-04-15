package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCanaryRoute_NoCanary(t *testing.T) {
	cd := NewCanaryDeploy()

	model, isCanary := cd.Route("llama3")
	if model != "llama3" {
		t.Errorf("expected llama3, got %q", model)
	}
	if isCanary {
		t.Error("expected isCanary=false when no canary configured")
	}
}

func TestCanaryRoute_TrafficSplit(t *testing.T) {
	cd := NewCanaryDeploy()
	cd.Configure(CanaryConfig{
		BaseModel:    "llama3",
		NewModel:     "llama3-v2",
		TrafficPct:   0.5,
		MaxErrorRate: 0.1,
	})

	canaryCount := 0
	baseCount := 0
	for i := 0; i < 1000; i++ {
		model, isCanary := cd.Route("llama3")
		if isCanary {
			canaryCount++
			if model != "llama3-v2" {
				t.Fatalf("canary should route to llama3-v2, got %q", model)
			}
		} else {
			baseCount++
			if model != "llama3" {
				t.Fatalf("base should route to llama3, got %q", model)
			}
		}
	}

	// With 50% split and 1000 requests, expect roughly 500 each.
	// Allow wide margin for randomness.
	if canaryCount < 350 || canaryCount > 650 {
		t.Errorf("unexpected traffic split: canary=%d, base=%d", canaryCount, baseCount)
	}
}

func TestCanaryRoute_DifferentModel(t *testing.T) {
	cd := NewCanaryDeploy()
	cd.Configure(CanaryConfig{
		BaseModel:    "llama3",
		NewModel:     "llama3-v2",
		TrafficPct:   1.0, // 100% canary
		MaxErrorRate: 0.1,
	})

	// Requests for a different model should pass through unchanged.
	model, isCanary := cd.Route("bert")
	if model != "bert" {
		t.Errorf("expected bert (unrelated model), got %q", model)
	}
	if isCanary {
		t.Error("unrelated model should not be marked as canary")
	}
}

func TestCanaryAutoRollback(t *testing.T) {
	cd := NewCanaryDeploy()
	cd.Configure(CanaryConfig{
		BaseModel:    "llama3",
		NewModel:     "llama3-v2",
		TrafficPct:   0.5,
		MaxErrorRate: 0.05, // 5%
	})

	// Simulate 11 requests: 10 errors + 1 success = ~91% error rate.
	// The rollback check requires at least 10 total requests.
	cd.RecordSuccess() // total=1
	for i := 0; i < 10; i++ {
		cd.RecordError() // total=2..11, errors=1..10
	}

	// Should have rolled back.
	status := cd.Status()
	if status == nil {
		t.Fatal("expected status, got nil")
	}
	if status.State != "rolled_back" {
		t.Errorf("expected state=rolled_back, got %q", status.State)
	}

	// After rollback, Route should return base model.
	model, isCanary := cd.Route("llama3")
	if model != "llama3" {
		t.Errorf("expected base model after rollback, got %q", model)
	}
	if isCanary {
		t.Error("should not be canary after rollback")
	}
}

func TestCanaryAutoPromote(t *testing.T) {
	cd := NewCanaryDeploy()
	cd.Configure(CanaryConfig{
		BaseModel:        "llama3",
		NewModel:         "llama3-v2",
		TrafficPct:       0.5,
		MaxErrorRate:     0.1,
		AutoPromoteAfter: 20,
	})

	// Simulate 20 successful requests.
	for i := 0; i < 20; i++ {
		cd.RecordSuccess()
	}

	status := cd.Status()
	if status == nil {
		t.Fatal("expected status, got nil")
	}
	if status.State != "promoted" {
		t.Errorf("expected state=promoted, got %q", status.State)
	}

	// After promotion, base model requests should route to new model.
	model, _ := cd.Route("llama3")
	if model != "llama3-v2" {
		t.Errorf("expected new model after promotion, got %q", model)
	}
}

func TestCanaryCancel(t *testing.T) {
	cd := NewCanaryDeploy()

	// Cancel with no active canary.
	if cd.Cancel() {
		t.Error("cancel should return false when no canary active")
	}

	cd.Configure(CanaryConfig{
		BaseModel:  "a",
		NewModel:   "b",
		TrafficPct: 0.5,
	})

	if !cd.Cancel() {
		t.Error("cancel should return true for active canary")
	}

	if cd.Status() != nil {
		t.Error("status should be nil after cancel")
	}
}

func TestHandleCanaryCreate(t *testing.T) {
	reg := NewRegistry()
	srv := NewServer(reg)
	canary := NewCanaryDeploy()
	srv.canary = canary

	body := `{"base_model":"llama3","new_model":"llama3-v2","traffic_pct":0.1,"max_error_rate":0.05,"auto_promote_after":100}`
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/canary", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var status CanaryStatus
	json.NewDecoder(rr.Body).Decode(&status)
	if status.State != "active" {
		t.Errorf("expected state=active, got %q", status.State)
	}
	if status.Config.TrafficPct != 0.1 {
		t.Errorf("expected traffic_pct=0.1, got %f", status.Config.TrafficPct)
	}
}

func TestHandleCanaryDelete(t *testing.T) {
	reg := NewRegistry()
	srv := NewServer(reg)
	canary := NewCanaryDeploy()
	srv.canary = canary

	canary.Configure(CanaryConfig{
		BaseModel:  "a",
		NewModel:   "b",
		TrafficPct: 0.5,
	})

	req := httptest.NewRequest(http.MethodDelete, "/v1/admin/canary", nil)
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	if canary.Status() != nil {
		t.Error("canary should be cancelled after DELETE")
	}
}
