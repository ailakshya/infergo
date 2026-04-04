package server_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ailakshya/infergo/server"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)


// TestInitTracer_NoEndpoint verifies that no-op tracing installs without error (T1).
func TestInitTracer_NoEndpoint(t *testing.T) {
	shutdown, err := server.InitTracer("infergo-test", "")
	if err != nil {
		t.Fatalf("InitTracer with empty endpoint: %v", err)
	}
	if shutdown == nil {
		t.Fatal("shutdown func must not be nil")
	}
	if err := shutdown(context.Background()); err != nil {
		t.Errorf("shutdown: %v", err)
	}
}

// TestWrapTracingMiddleware verifies spans are emitted for handled requests (T2).
func TestWrapTracingMiddleware(t *testing.T) {
	// Install an in-memory exporter so we can inspect spans.
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { tp.Shutdown(context.Background()) })

	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	wrapped := server.WrapTracing(inner, "test-op")

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/v1/models", nil)
	wrapped.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rec.Code)
	}

	// Flush and check that at least one span was recorded.
	tp.ForceFlush(context.Background())
	spans := exporter.GetSpans()
	if len(spans) == 0 {
		t.Error("expected at least one span to be emitted")
	}
}

// TestTracePropagation verifies W3C traceparent header causes child span to link correctly (T2).
func TestTracePropagation(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { tp.Shutdown(context.Background()) })

	// Install W3C propagator via InitTracer no-op path, then restore the recording provider.
	server.InitTracer("infergo-test", "")
	otel.SetTracerProvider(tp)

	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	wrapped := server.WrapTracing(inner, "test-op")

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/v1/models", nil)
	// Inject a W3C traceparent — otelhttp will link the created span as a child.
	req.Header.Set("Traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
	wrapped.ServeHTTP(rec, req)

	tp.ForceFlush(context.Background())
	spans := exporter.GetSpans()
	if len(spans) == 0 {
		t.Fatal("expected at least one span to be emitted")
	}

	// The span's parent should be the remote span from the traceparent header.
	parentTraceID := spans[0].SpanContext.TraceID().String()
	if parentTraceID != "4bf92f3577b34da6a3ce929d0e0e4736" {
		t.Errorf("trace propagation: got trace ID %q, want 4bf92f3577b34da6a3ce929d0e0e4736", parentTraceID)
	}
}

// TestNoOpTracing_NoOverhead verifies a no-op provider returns 200 without panic (T4).
func TestNoOpTracing_NoOverhead(t *testing.T) {
	if _, err := server.InitTracer("infergo-test", ""); err != nil {
		t.Fatalf("InitTracer: %v", err)
	}

	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	wrapped := server.WrapTracing(inner, "no-op-test")

	for i := 0; i < 100; i++ {
		rec := httptest.NewRecorder()
		wrapped.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("iteration %d: expected 200, got %d", i, rec.Code)
		}
	}
}
