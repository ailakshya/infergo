package server

import (
	"context"
	"net/http"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace/noop"
)

// InitTracer sets up the global OpenTelemetry trace provider.
//
// If otlpEndpoint is empty, a no-op tracer is installed and all tracing calls
// become zero-cost. If an endpoint is given (e.g. "localhost:4318"), spans are
// exported via OTLP HTTP to that address.
//
// The returned shutdown function must be called on program exit to flush the
// exporter. It is safe to call even when tracing is disabled.
func InitTracer(serviceName, otlpEndpoint string) (shutdown func(context.Context) error, err error) {
	if otlpEndpoint == "" {
		// No-op: install a provider that does nothing.
		otel.SetTracerProvider(noop.NewTracerProvider())
		otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
			propagation.TraceContext{},
			propagation.Baggage{},
		))
		return func(context.Context) error { return nil }, nil
	}

	exp, err := otlptracehttp.New(
		context.Background(),
		otlptracehttp.WithEndpoint(otlpEndpoint),
		otlptracehttp.WithInsecure(),
	)
	if err != nil {
		return nil, err
	}

	res, err := resource.New(
		context.Background(),
		resource.WithAttributes(semconv.ServiceName(serviceName)),
	)
	if err != nil {
		return nil, err
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
	)

	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	return tp.Shutdown, nil
}

// WrapTracing wraps an http.Handler with OpenTelemetry span instrumentation.
//
// Each request creates a span with http.method, http.route, and http.status_code
// attributes. W3C traceparent headers are read from incoming requests and
// propagated to outgoing context.
//
// When tracing is disabled (no-op provider installed by InitTracer), this adds
// no measurable overhead.
func WrapTracing(handler http.Handler, operationName string) http.Handler {
	return otelhttp.NewHandler(handler, operationName)
}
