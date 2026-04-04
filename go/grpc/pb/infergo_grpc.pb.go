// Package pb contains the gRPC service interface and registration helpers for
// the Infergo API. These are hand-written equivalents of protoc-gen-go-grpc
// output. The service uses a JSON codec (see grpc package) rather than
// protobuf wire format to avoid a protoc dependency.
package pb

import (
	"context"

	"google.golang.org/grpc"
)

// ─── Service description constants ───────────────────────────────────────────

const (
	Infergo_ChatCompletion_FullMethodName = "/infergo.v1.Infergo/ChatCompletion"
	Infergo_Embed_FullMethodName          = "/infergo.v1.Infergo/Embed"
	Infergo_Detect_FullMethodName         = "/infergo.v1.Infergo/Detect"
	Infergo_ListModels_FullMethodName     = "/infergo.v1.Infergo/ListModels"
)

// ─── Server-side streaming interface ─────────────────────────────────────────

// InfergoServer is the interface that must be implemented by a gRPC server.
type InfergoServer interface {
	// ChatCompletion streams tokens back to the client.
	ChatCompletion(*ChatRequest, Infergo_ChatCompletionServer) error
	// Embed returns a float32 embedding vector.
	Embed(context.Context, *EmbedRequest) (*EmbedResponse, error)
	// Detect runs object detection and returns bounding boxes.
	Detect(context.Context, *DetectRequest) (*DetectResponse, error)
	// ListModels returns all registered models.
	ListModels(context.Context, *ListModelsRequest) (*ListModelsResponse, error)
	mustEmbedUnimplementedInfergoServer()
}

// Infergo_ChatCompletionServer is used by the server to stream ChatChunks.
type Infergo_ChatCompletionServer interface {
	Send(*ChatChunk) error
	grpc.ServerStream
}

type infergo_ChatCompletionServer struct {
	grpc.ServerStream
}

func (s *infergo_ChatCompletionServer) Send(m *ChatChunk) error {
	return s.ServerStream.SendMsg(m)
}

// ─── Unimplemented server (embed for forward compatibility) ──────────────────

// UnimplementedInfergoServer must be embedded in server implementations.
type UnimplementedInfergoServer struct{}

func (UnimplementedInfergoServer) ChatCompletion(*ChatRequest, Infergo_ChatCompletionServer) error {
	return nil
}
func (UnimplementedInfergoServer) Embed(context.Context, *EmbedRequest) (*EmbedResponse, error) {
	return nil, nil
}
func (UnimplementedInfergoServer) Detect(context.Context, *DetectRequest) (*DetectResponse, error) {
	return nil, nil
}
func (UnimplementedInfergoServer) ListModels(context.Context, *ListModelsRequest) (*ListModelsResponse, error) {
	return nil, nil
}
func (UnimplementedInfergoServer) mustEmbedUnimplementedInfergoServer() {}

// ─── Client-side interface ────────────────────────────────────────────────────

// InfergoClient is the client API for the Infergo service.
type InfergoClient interface {
	ChatCompletion(ctx context.Context, in *ChatRequest, opts ...grpc.CallOption) (Infergo_ChatCompletionClient, error)
	Embed(ctx context.Context, in *EmbedRequest, opts ...grpc.CallOption) (*EmbedResponse, error)
	Detect(ctx context.Context, in *DetectRequest, opts ...grpc.CallOption) (*DetectResponse, error)
	ListModels(ctx context.Context, in *ListModelsRequest, opts ...grpc.CallOption) (*ListModelsResponse, error)
}

// Infergo_ChatCompletionClient is used by the client to receive streamed ChatChunks.
type Infergo_ChatCompletionClient interface {
	Recv() (*ChatChunk, error)
	grpc.ClientStream
}

type infergo_ChatCompletionClient struct {
	grpc.ClientStream
}

func (c *infergo_ChatCompletionClient) Recv() (*ChatChunk, error) {
	m := new(ChatChunk)
	if err := c.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

// ─── Client implementation ────────────────────────────────────────────────────

type infergoClient struct {
	cc grpc.ClientConnInterface
}

// NewInfergoClient creates a new gRPC client connected to cc.
func NewInfergoClient(cc grpc.ClientConnInterface) InfergoClient {
	return &infergoClient{cc}
}

func (c *infergoClient) ChatCompletion(ctx context.Context, in *ChatRequest, opts ...grpc.CallOption) (Infergo_ChatCompletionClient, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &Infergo_ServiceDesc.Streams[0], Infergo_ChatCompletion_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &infergo_ChatCompletionClient{stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

func (c *infergoClient) Embed(ctx context.Context, in *EmbedRequest, opts ...grpc.CallOption) (*EmbedResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(EmbedResponse)
	err := c.cc.Invoke(ctx, Infergo_Embed_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *infergoClient) Detect(ctx context.Context, in *DetectRequest, opts ...grpc.CallOption) (*DetectResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(DetectResponse)
	err := c.cc.Invoke(ctx, Infergo_Detect_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *infergoClient) ListModels(ctx context.Context, in *ListModelsRequest, opts ...grpc.CallOption) (*ListModelsResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(ListModelsResponse)
	err := c.cc.Invoke(ctx, Infergo_ListModels_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ─── Server registration ──────────────────────────────────────────────────────

// RegisterInfergoServer registers the InfergoServer implementation with a gRPC server.
func RegisterInfergoServer(s grpc.ServiceRegistrar, srv InfergoServer) {
	s.RegisterService(&Infergo_ServiceDesc, srv)
}

func _Infergo_ChatCompletion_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(ChatRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(InfergoServer).ChatCompletion(m, &infergo_ChatCompletionServer{stream})
}

func _Infergo_Embed_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(EmbedRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(InfergoServer).Embed(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Infergo_Embed_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(InfergoServer).Embed(ctx, req.(*EmbedRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Infergo_Detect_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DetectRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(InfergoServer).Detect(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Infergo_Detect_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(InfergoServer).Detect(ctx, req.(*DetectRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Infergo_ListModels_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ListModelsRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(InfergoServer).ListModels(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Infergo_ListModels_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(InfergoServer).ListModels(ctx, req.(*ListModelsRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// Infergo_ServiceDesc describes the Infergo service for gRPC registration.
var Infergo_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "infergo.v1.Infergo",
	HandlerType: (*InfergoServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Embed",
			Handler:    _Infergo_Embed_Handler,
		},
		{
			MethodName: "Detect",
			Handler:    _Infergo_Detect_Handler,
		},
		{
			MethodName: "ListModels",
			Handler:    _Infergo_ListModels_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "ChatCompletion",
			Handler:       _Infergo_ChatCompletion_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "infergo.proto",
}
