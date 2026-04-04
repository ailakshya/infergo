package main

import (
	infergogrpc "github.com/ailakshya/infergo/grpc"
	"github.com/ailakshya/infergo/server"
)

// serverRegistryAdapter wraps *server.Registry to satisfy infergogrpc.ModelRegistry.
// It lives in cmd/infergo (not in the grpc package) so that the grpc package
// remains free of the CGo dependency that server/batcher.go → tensor introduces.
type serverRegistryAdapter struct {
	reg *server.Registry
}

// newGRPCRegistry wraps a *server.Registry for use with infergogrpc.New().
func newGRPCRegistry(reg *server.Registry) infergogrpc.ModelRegistry {
	return &serverRegistryAdapter{reg: reg}
}

type serverModelHandle struct {
	ref *server.ModelRef
}

func (h *serverModelHandle) Model() interface{} { return h.ref.Model }
func (h *serverModelHandle) Release()           { h.ref.Release() }

func (a *serverRegistryAdapter) Get(name string) (infergogrpc.ModelHandle, error) {
	ref, err := a.reg.Get(name)
	if err != nil {
		return nil, err
	}
	return &serverModelHandle{ref: ref}, nil
}

func (a *serverRegistryAdapter) Names() []string {
	return a.reg.Names()
}
