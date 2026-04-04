// Package pb contains the gRPC message types for the Infergo API.
// These types correspond to proto/infergo.proto but are hand-written
// to avoid a protoc dependency. The gRPC server uses a JSON codec so
// standard encoding/json struct tags are used instead of protobuf wire format.
package pb

// Message is a single chat turn (role + content).
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest is sent by the client to start a streaming chat completion.
type ChatRequest struct {
	Model       string     `json:"model"`
	Messages    []*Message `json:"messages"`
	MaxTokens   int32      `json:"max_tokens"`
	Temperature float32    `json:"temperature"`
}

// ChatChunk is a single streamed token returned by ChatCompletion.
type ChatChunk struct {
	Token string `json:"token"`
	Done  bool   `json:"done"`
}

// EmbedRequest asks the server to embed a text input.
type EmbedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// EmbedResponse holds the embedding vector.
type EmbedResponse struct {
	Values []float32 `json:"values"`
}

// DetectRequest asks the server to run object detection.
type DetectRequest struct {
	Model      string  `json:"model"`
	ImageBytes []byte  `json:"image_bytes"`
	ConfThresh float32 `json:"conf_thresh"`
	IouThresh  float32 `json:"iou_thresh"`
}

// Detection is a single detected object.
type Detection struct {
	X1         float32 `json:"x1"`
	Y1         float32 `json:"y1"`
	X2         float32 `json:"x2"`
	Y2         float32 `json:"y2"`
	ClassId    int32   `json:"class_id"`
	Confidence float32 `json:"confidence"`
}

// DetectResponse holds all detections from a DetectRequest.
type DetectResponse struct {
	Detections []*Detection `json:"detections"`
}

// ListModelsRequest has no fields.
type ListModelsRequest struct{}

// ModelInfo describes a single registered model.
type ModelInfo struct {
	Name string `json:"name"`
}

// ListModelsResponse lists all available models.
type ListModelsResponse struct {
	Models []*ModelInfo `json:"models"`
}
