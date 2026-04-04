package server

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds all Prometheus metrics for the infergo server.
type Metrics struct {
	registry *prometheus.Registry

	// infergo_requests_total: counter by model, endpoint, status class
	RequestsTotal *prometheus.CounterVec

	// infergo_request_duration_seconds: histogram by model, endpoint
	RequestDuration *prometheus.HistogramVec

	// infergo_batch_size: histogram of batch sizes dispatched by the batcher
	BatchSize prometheus.Histogram

	// infergo_tokens_per_second: gauge by model (updated after each generation)
	TokensPerSecond *prometheus.GaugeVec

	// infergo_gpu_memory_bytes: gauge by device id
	GPUMemoryBytes *prometheus.GaugeVec

	// infergo_queue_depth: gauge of total in-flight requests (active + waiting)
	QueueDepth prometheus.Gauge
}

// NewMetrics creates and registers all metrics on a fresh Prometheus registry.
func NewMetrics() *Metrics {
	reg := prometheus.NewRegistry()

	m := &Metrics{
		registry: reg,

		RequestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "infergo_requests_total",
			Help: "Total number of inference requests, by model, endpoint, and HTTP status class.",
		}, []string{"model", "endpoint", "status"}),

		RequestDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "infergo_request_duration_seconds",
			Help:    "Inference request latency in seconds, by model and endpoint.",
			Buckets: prometheus.DefBuckets,
		}, []string{"model", "endpoint"}),

		BatchSize: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "infergo_batch_size",
			Help:    "Number of requests dispatched per batch.",
			Buckets: []float64{1, 2, 4, 8, 16, 32, 64, 128},
		}),

		TokensPerSecond: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "infergo_tokens_per_second",
			Help: "Tokens generated per second for the most recent request, by model.",
		}, []string{"model"}),

		GPUMemoryBytes: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "infergo_gpu_memory_bytes",
			Help: "GPU device memory in use (bytes), by device id.",
		}, []string{"device"}),

		QueueDepth: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "infergo_queue_depth",
			Help: "Current number of requests in the priority queue (active + waiting).",
		}),
	}

	reg.MustRegister(
		m.RequestsTotal,
		m.RequestDuration,
		m.BatchSize,
		m.TokensPerSecond,
		m.GPUMemoryBytes,
		m.QueueDepth,
		prometheus.NewGoCollector(),
		prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}),
	)

	return m
}

// Handler returns an HTTP handler that serves the Prometheus text exposition.
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{})
}

// ObserveBatch records a batch size observation.
func (m *Metrics) ObserveBatch(size int) {
	m.BatchSize.Observe(float64(size))
}

// ObserveTokensPerSecond records the tokens/s for a model after generation.
func (m *Metrics) ObserveTokensPerSecond(model string, tokens int, elapsed time.Duration) {
	if elapsed > 0 {
		m.TokensPerSecond.WithLabelValues(model).Set(
			float64(tokens) / elapsed.Seconds(),
		)
	}
}

// SetGPUMemory records current GPU memory usage for a device.
func (m *Metrics) SetGPUMemory(deviceID int, bytes int64) {
	m.GPUMemoryBytes.WithLabelValues(strconv.Itoa(deviceID)).Set(float64(bytes))
}

// ─── Middleware ───────────────────────────────────────────────────────────────

// responseRecorder captures the status code written by a handler.
// It forwards Flush() to the underlying ResponseWriter when available,
// so SSE streaming works through the metrics middleware.
type responseRecorder struct {
	http.ResponseWriter
	status int
}

func (r *responseRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// Flush implements http.Flusher by forwarding to the underlying writer.
func (r *responseRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (r *responseRecorder) statusClass() string {
	switch {
	case r.status < 300:
		return "2xx"
	case r.status < 400:
		return "3xx"
	case r.status < 500:
		return "4xx"
	default:
		return "5xx"
	}
}

// InstrumentHandler wraps an http.Handler to record request metrics.
// model and endpoint are the label values for this route.
func (m *Metrics) InstrumentHandler(model, endpoint string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := &responseRecorder{ResponseWriter: w, status: http.StatusOK}
		start := time.Now()
		next.ServeHTTP(rec, r)
		elapsed := time.Since(start)

		status := rec.statusClass()
		m.RequestsTotal.WithLabelValues(model, endpoint, status).Inc()
		m.RequestDuration.WithLabelValues(model, endpoint).Observe(elapsed.Seconds())
	})
}

// WrapServer returns a new http.Handler that instruments every request to srv.
// The model label is extracted from the "model" JSON field when available;
// for routes where the model is unknown at routing time, "_unknown" is used.
func (m *Metrics) WrapServer(srv http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := &responseRecorder{ResponseWriter: w, status: http.StatusOK}
		start := time.Now()
		srv.ServeHTTP(rec, r)
		elapsed := time.Since(start)

		status := rec.statusClass()
		m.RequestsTotal.WithLabelValues("_all", r.URL.Path, status).Inc()
		m.RequestDuration.WithLabelValues("_all", r.URL.Path).Observe(elapsed.Seconds())
	})
}
