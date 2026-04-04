package grpc

import (
	"context"
	"fmt"
	"net"
	"strings"

	"google.golang.org/grpc"

	"github.com/ailakshya/infergo/grpc/pb"
)

// ─── Registry and model interfaces ───────────────────────────────────────────
// These mirror the corresponding types in server/ without importing that package
// (which has a CGo dependency via tensor). Any *server.Registry satisfies
// ModelRegistry automatically because it has the same Get/Names methods.

// ModelHandle is a reference to a loaded model. The caller must call Release
// when the model is no longer needed.
type ModelHandle interface {
	// Model returns the underlying model value.
	Model() interface{}
	// Release decrements the reference count.
	Release()
}

// ModelRegistry is satisfied by *server.Registry (duck-typed).
type ModelRegistry interface {
	// Get returns a handle for the named model or an error if not found.
	Get(name string) (ModelHandle, error)
	// Names returns the names of all currently registered models.
	Names() []string
}

// LLMModel is a model that supports text generation.
type LLMModel interface {
	Generate(ctx context.Context, prompt string, maxTokens int, temp float32) (text string, promptToks int, genToks int, err error)
}

// EmbeddingModel is a model that produces embedding vectors.
type EmbeddingModel interface {
	Embed(ctx context.Context, input string) ([]float32, error)
}

// DetectedObject is a single detection result.
type DetectedObject struct {
	X1, Y1, X2, Y2 float32
	ClassID         int
	Confidence      float32
}

// DetectionModel is a model that runs object detection.
type DetectionModel interface {
	Detect(ctx context.Context, imageBytes []byte, confThresh, iouThresh float32) ([]DetectedObject, error)
}

// ─── Server ───────────────────────────────────────────────────────────────────

// Server wraps a ModelRegistry and serves gRPC traffic using a JSON codec.
// It implements ChatCompletion (server-streaming), Embed, Detect, and
// ListModels — mirroring the HTTP API in go/server/router.go.
type Server struct {
	pb.UnimplementedInfergoServer
	reg     ModelRegistry
	grpcSrv *grpc.Server
}

// New creates a gRPC Server backed by the given registry.
// The JSON codec is registered automatically via the package init() in codec.go.
func New(reg ModelRegistry) *Server {
	s := &Server{reg: reg}
	s.grpcSrv = grpc.NewServer(
		grpc.ForceServerCodec(JSONCodec{}),
	)
	pb.RegisterInfergoServer(s.grpcSrv, s)
	return s
}

// Serve starts listening on addr (e.g. ":9091") and blocks until the server
// stops or encounters an unrecoverable error.
func (s *Server) Serve(addr string) error {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("grpc: listen %s: %w", addr, err)
	}
	return s.grpcSrv.Serve(lis)
}

// Stop gracefully stops the gRPC server, waiting for active RPCs to finish.
func (s *Server) Stop() { s.grpcSrv.GracefulStop() }

// ─── ChatCompletion (server-streaming) ───────────────────────────────────────

func (s *Server) ChatCompletion(req *pb.ChatRequest, stream pb.Infergo_ChatCompletionServer) error {
	if req.Model == "" {
		return fmt.Errorf("model field is required")
	}
	if len(req.Messages) == 0 {
		return fmt.Errorf("messages must not be empty")
	}

	handle, err := s.reg.Get(req.Model)
	if err != nil {
		return fmt.Errorf("model not found: %w", err)
	}
	defer handle.Release()

	llm, ok := handle.Model().(LLMModel)
	if !ok {
		return fmt.Errorf("model %q does not support chat completions", req.Model)
	}

	msgs := make([]chatMsg, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = chatMsg{Role: m.Role, Content: m.Content}
	}
	prompt := buildPrompt(msgs)

	maxToks := int(req.MaxTokens)
	if maxToks <= 0 {
		maxToks = 256
	}

	text, _, _, err := llm.Generate(stream.Context(), prompt, maxToks, req.Temperature)
	if err != nil {
		return fmt.Errorf("generation failed: %w", err)
	}

	// Send the full text as a single content chunk, then a done marker.
	if err := stream.Send(&pb.ChatChunk{Token: text, Done: false}); err != nil {
		return err
	}
	return stream.Send(&pb.ChatChunk{Token: "", Done: true})
}

// ─── Embed ────────────────────────────────────────────────────────────────────

func (s *Server) Embed(ctx context.Context, req *pb.EmbedRequest) (*pb.EmbedResponse, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("model field is required")
	}
	if req.Input == "" {
		return nil, fmt.Errorf("input must not be empty")
	}

	handle, err := s.reg.Get(req.Model)
	if err != nil {
		return nil, fmt.Errorf("model not found: %w", err)
	}
	defer handle.Release()

	emb, ok := handle.Model().(EmbeddingModel)
	if !ok {
		return nil, fmt.Errorf("model %q does not support embeddings", req.Model)
	}

	vec, err := emb.Embed(ctx, req.Input)
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}

	return &pb.EmbedResponse{Values: vec}, nil
}

// ─── Detect ───────────────────────────────────────────────────────────────────

func (s *Server) Detect(ctx context.Context, req *pb.DetectRequest) (*pb.DetectResponse, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("model field is required")
	}
	if len(req.ImageBytes) == 0 {
		return nil, fmt.Errorf("image_bytes must not be empty")
	}

	handle, err := s.reg.Get(req.Model)
	if err != nil {
		return nil, fmt.Errorf("model not found: %w", err)
	}
	defer handle.Release()

	det, ok := handle.Model().(DetectionModel)
	if !ok {
		return nil, fmt.Errorf("model %q does not support detection", req.Model)
	}

	confThresh := req.ConfThresh
	if confThresh == 0 {
		confThresh = 0.25
	}
	iouThresh := req.IouThresh
	if iouThresh == 0 {
		iouThresh = 0.45
	}

	objs, err := det.Detect(ctx, req.ImageBytes, confThresh, iouThresh)
	if err != nil {
		return nil, fmt.Errorf("detection failed: %w", err)
	}

	dets := make([]*pb.Detection, len(objs))
	for i, o := range objs {
		dets[i] = &pb.Detection{
			X1:         o.X1,
			Y1:         o.Y1,
			X2:         o.X2,
			Y2:         o.Y2,
			ClassId:    int32(o.ClassID),
			Confidence: o.Confidence,
		}
	}
	return &pb.DetectResponse{Detections: dets}, nil
}

// ─── ListModels ───────────────────────────────────────────────────────────────

func (s *Server) ListModels(_ context.Context, _ *pb.ListModelsRequest) (*pb.ListModelsResponse, error) {
	names := s.reg.Names()
	models := make([]*pb.ModelInfo, len(names))
	for i, n := range names {
		models[i] = &pb.ModelInfo{Name: n}
	}
	return &pb.ListModelsResponse{Models: models}, nil
}

// ─── helpers ─────────────────────────────────────────────────────────────────

type chatMsg struct {
	Role    string
	Content string
}

// buildPrompt flattens a message list into a plain text prompt.
func buildPrompt(messages []chatMsg) string {
	var out string
	for _, m := range messages {
		switch m.Role {
		case "system":
			out += "[system]: " + m.Content + "\n"
		case "user":
			out += "[user]: " + m.Content + "\n"
		case "assistant":
			out += "[assistant]: " + m.Content + "\n"
		default:
			out += m.Content + "\n"
		}
	}
	return strings.TrimRight(out, "\n")
}
