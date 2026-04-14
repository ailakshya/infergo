// Direct ggml kernel invocation — use llama.cpp's perfectly tuned CUDA kernels
// without the ggml graph overhead.
//
// Strategy: create minimal ggml tensors pointing to our GPU buffers,
// then call ggml_backend_cuda operations directly.

#include "engine.cuh"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cstdio>

namespace infergo {
namespace cuda {

// Wrapper that uses ggml's CUDA backend for matmul
// but skips the graph construction overhead
struct GGMLDirect {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;

    bool Init() {
        backend = ggml_backend_cuda_init(0);
        if (!backend) {
            printf("[ggml_direct] CUDA backend init failed\n");
            return false;
        }
        buft = ggml_backend_get_default_buffer_type(backend);
        return true;
    }

    // Create a ggml tensor wrapping existing GPU memory
    // WARNING: the tensor doesn't own the memory
    ggml_tensor* WrapGPU(ggml_context* ctx, const char* name,
                          void* data, ggml_type type,
                          int64_t ne0, int64_t ne1 = 1) {
        ggml_tensor* t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        ggml_set_name(t, name);
        t->data = data;
        return t;
    }

    // Compute matmul: out = weight @ input
    // Uses ggml's optimized CUDA kernels (mmvq for quantized, cublas for FP16)
    void MatMul(ggml_context* ctx,
                void* weight_data, ggml_type weight_type, int64_t in_dim, int64_t out_dim,
                void* input_data, void* output_data) {
        // Build minimal graph
        ggml_tensor* weight = WrapGPU(ctx, "w", weight_data, weight_type, in_dim, out_dim);
        ggml_tensor* input = WrapGPU(ctx, "x", input_data, GGML_TYPE_F16, in_dim, 1);
        ggml_tensor* output = ggml_mul_mat(ctx, weight, input);
        output->data = output_data;

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);

        ggml_backend_graph_compute(backend, graph);
    }

    ~GGMLDirect() {
        if (backend) ggml_backend_free(backend);
    }
};

} // namespace cuda
} // namespace infergo
