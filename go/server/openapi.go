package server

import (
	"encoding/json"
	"net/http"
)

// OpenAPI 3.0 spec — auto-generated from registered endpoints.
// Serves at GET /v1/openapi.json and GET /ui/docs (Swagger UI).

func (s *Server) handleOpenAPISpec(w http.ResponseWriter, r *http.Request) {
	spec := map[string]interface{}{
		"openapi": "3.0.3",
		"info": map[string]interface{}{
			"title":       "infergo API",
			"description": "Production AI inference platform. LLM + Embedding + Detection + RAG in one binary.",
			"version":     "1.1.0",
			"license": map[string]string{
				"name": "Apache 2.0",
				"url":  "https://www.apache.org/licenses/LICENSE-2.0",
			},
		},
		"servers": []map[string]string{
			{"url": "http://localhost:9090", "description": "Local server"},
		},
		"paths": map[string]interface{}{
			"/v1/chat/completions": map[string]interface{}{
				"post": endpoint("Chat Completion", "Generate chat completion with optional streaming, JSON mode, function calling, grammar constraints.",
					reqBody("ChatCompletionRequest", map[string]interface{}{
						"model":    prop("string", "Model identifier", true),
						"messages": prop("array", "Chat messages", true),
						"max_tokens": prop("integer", "Max tokens to generate", false),
						"temperature": prop("number", "Sampling temperature 0-2", false),
						"stream":   prop("boolean", "Stream response via SSE", false),
						"response_format": prop("object", "Force output format: json_object, toon, grammar", false),
						"tools":    prop("array", "Function definitions for tool calling", false),
					}),
					respBody("ChatCompletionResponse")),
			},
			"/v1/completions": map[string]interface{}{
				"post": endpoint("Text Completion", "Generate text completion from a prompt.",
					reqBody("CompletionRequest", map[string]interface{}{
						"model":  prop("string", "Model identifier", true),
						"prompt": prop("string", "Text prompt", true),
						"max_tokens": prop("integer", "Max tokens", false),
					}),
					respBody("CompletionResponse")),
			},
			"/v1/embeddings": map[string]interface{}{
				"post": endpoint("Embeddings", "Generate dense vector embeddings for text.",
					reqBody("EmbeddingRequest", map[string]interface{}{
						"model": prop("string", "Embedding model", true),
						"input": prop("string", "Text or array of texts", true),
					}),
					respBody("EmbeddingResponse")),
			},
			"/v1/search": map[string]interface{}{
				"post": endpoint("Vector Search", "Search indexed documents by semantic similarity.",
					reqBody("SearchRequest", map[string]interface{}{
						"model": prop("string", "Embedding model for query", true),
						"query": prop("string", "Search query", true),
						"k":     prop("integer", "Number of results", false),
						"mode":  prop("string", "Search mode: vector, bm25, hybrid", false),
					}),
					respBody("SearchResponse")),
			},
			"/v1/rerank": map[string]interface{}{
				"post": endpoint("Rerank", "Rerank documents by relevance to query.",
					reqBody("RerankRequest", map[string]interface{}{
						"model":     prop("string", "Embedding model", true),
						"query":     prop("string", "Query text", true),
						"documents": prop("array", "Documents to rerank", true),
						"top_n":     prop("integer", "Return top N", false),
					}),
					respBody("RerankResponse")),
			},
			"/v1/rag": map[string]interface{}{
				"post": endpoint("RAG Pipeline", "End-to-end retrieval-augmented generation.",
					reqBody("RAGRequest", map[string]interface{}{
						"model":       prop("string", "LLM model", true),
						"embed_model": prop("string", "Embedding model", true),
						"query":       prop("string", "User question", true),
						"k":           prop("integer", "Number of docs to retrieve", false),
					}),
					respBody("RAGResponse")),
			},
			"/v1/detect": map[string]interface{}{
				"post": endpoint("Object Detection (JSON)", "Detect objects in a base64-encoded image.",
					reqBody("DetectRequest", map[string]interface{}{
						"model":       prop("string", "Detection model", true),
						"image_b64":   prop("string", "Base64 JPEG image", true),
						"conf_thresh": prop("number", "Confidence threshold", false),
						"iou_thresh":  prop("number", "NMS IoU threshold", false),
						"max_det":     prop("integer", "Max detections", false),
						"classes":     prop("array", "Filter class IDs", false),
					}),
					respBody("DetectResponse")),
			},
			"/v1/detect/binary": map[string]interface{}{
				"post": endpoint("Object Detection (Binary)", "Detect objects from raw JPEG bytes. Faster than JSON endpoint.",
					nil, respBody("DetectResponse")),
			},
			"/v1/ingest": map[string]interface{}{
				"post": endpoint("Document Ingestion", "Ingest documents into vector DB for RAG.",
					reqBody("IngestRequest", map[string]interface{}{
						"model": prop("string", "Embedding model", true),
						"texts": prop("array", "Text documents to ingest", true),
					}),
					respBody("IngestResponse")),
			},
			"/v1/models": map[string]interface{}{
				"get": endpoint("List Models", "List all loaded models.", nil, respBody("ModelListResponse")),
			},
			"/v1/audio/transcriptions": map[string]interface{}{
				"post": endpoint("Speech to Text", "Transcribe audio to text (Whisper).", nil, respBody("TranscriptionResponse")),
			},
			"/v1/batches": map[string]interface{}{
				"post": endpoint("Create Batch", "Submit async batch inference job.",
					reqBody("BatchRequest", map[string]interface{}{
						"model":   prop("string", "Model", true),
						"prompts": prop("array", "Prompts to process", true),
					}),
					respBody("BatchResponse")),
				"get": endpoint("Batch Status", "Check batch job status.", nil, respBody("BatchStatus")),
			},
			"/v1/admin/reload": map[string]interface{}{
				"post": endpoint("Hot Reload", "Hot-swap model weights without restart.", nil, nil),
			},
			"/v1/feedback": map[string]interface{}{
				"post": endpoint("Submit Feedback", "Rate a response for quality tracking.",
					reqBody("FeedbackRequest", map[string]interface{}{
						"request_id": prop("string", "Original request ID", true),
						"rating":     prop("string", "positive or negative", true),
						"comment":    prop("string", "Optional comment", false),
					}), nil),
			},
			"/health/live": map[string]interface{}{
				"get": endpoint("Liveness", "Kubernetes liveness probe.", nil, nil),
			},
			"/health/ready": map[string]interface{}{
				"get": endpoint("Readiness", "Kubernetes readiness probe.", nil, nil),
			},
			"/metrics": map[string]interface{}{
				"get": endpoint("Prometheus Metrics", "Prometheus-compatible metrics.", nil, nil),
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(spec)
}

// handleSwaggerUI serves a minimal Swagger UI page.
func (s *Server) handleSwaggerUI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(`<!DOCTYPE html>
<html><head>
<title>infergo API Docs</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head><body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({url:"/v1/openapi.json",dom_id:"#swagger-ui",deepLinking:true})</script>
</body></html>`))
}

// helpers for spec construction
func endpoint(summary, desc string, req, resp interface{}) map[string]interface{} {
	e := map[string]interface{}{
		"summary":     summary,
		"description": desc,
		"tags":        []string{summary},
	}
	if req != nil {
		e["requestBody"] = req
	}
	if resp != nil {
		e["responses"] = map[string]interface{}{
			"200": resp,
		}
	} else {
		e["responses"] = map[string]interface{}{
			"200": map[string]interface{}{"description": "OK"},
		}
	}
	return e
}

func reqBody(name string, props map[string]interface{}) map[string]interface{} {
	return map[string]interface{}{
		"required": true,
		"content": map[string]interface{}{
			"application/json": map[string]interface{}{
				"schema": map[string]interface{}{
					"type":       "object",
					"properties": props,
				},
			},
		},
	}
}

func respBody(name string) map[string]interface{} {
	return map[string]interface{}{
		"description": "Success",
		"content": map[string]interface{}{
			"application/json": map[string]interface{}{
				"schema": map[string]interface{}{"type": "object"},
			},
		},
	}
}

func prop(typ, desc string, required bool) map[string]interface{} {
	return map[string]interface{}{
		"type":        typ,
		"description": desc,
	}
}
