// Package grpc provides a gRPC server for the Infergo API.
// It uses a JSON codec over HTTP/2 so that no protoc installation is required.
package grpc

import (
	"encoding/json"

	"google.golang.org/grpc/encoding"
)

const jsonCodecName = "json"

func init() {
	encoding.RegisterCodec(JSONCodec{})
}

// JSONCodec is a gRPC codec that marshals messages as JSON rather than
// protobuf binary. Both client and server must use the same codec.
type JSONCodec struct{}

func (JSONCodec) Name() string { return jsonCodecName }

func (JSONCodec) Marshal(v interface{}) ([]byte, error) {
	return json.Marshal(v)
}

func (JSONCodec) Unmarshal(data []byte, v interface{}) error {
	return json.Unmarshal(data, v)
}
