// GGUF model loader — parses GGUF v3 format and uploads Q4_K weights to GPU
#include "engine.cuh"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace infergo {
namespace cuda {

// ─── GGUF format constants ──────────────────────────────────────────────────

enum GGUFValueType {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

enum GGMLType {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
};

// Bytes per element for each type (for block types: bytes per block / elements per block)
static size_t ggml_type_block_size(int type) {
    switch (type) {
        case GGML_TYPE_F32:  return 1;    // 4 bytes per element
        case GGML_TYPE_F16:  return 1;    // 2 bytes per element
        case GGML_TYPE_Q4_K: return 256;  // 256 elements per block
        case GGML_TYPE_Q6_K: return 256;
        default: return 256;
    }
}

static size_t ggml_type_bytes_per_block(int type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q4_K: return 144;  // sizeof Q4_K block
        case GGML_TYPE_Q6_K: return 210;
        default: return 144;
    }
}

struct TensorInfo {
    std::string name;
    int type;
    int n_dims;
    int64_t dims[4];
    uint64_t offset;
    size_t total_elements;
    size_t data_size;
};

// ─── GGUF Reader ─────────────────────────────────────────────────────────────

class GGUFReader {
public:
    bool Open(const char* path) {
        file_.open(path, std::ios::binary);
        if (!file_.is_open()) return false;

        // Read header
        uint32_t magic;
        file_.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != 0x46554747) { // "GGUF"
            printf("[gguf] Invalid magic: 0x%08X\n", magic);
            return false;
        }

        file_.read(reinterpret_cast<char*>(&version_), 4);
        file_.read(reinterpret_cast<char*>(&n_tensors_), 8);
        file_.read(reinterpret_cast<char*>(&n_kv_), 8);

        printf("[gguf] v%d, %llu tensors, %llu kv pairs\n",
               version_, (unsigned long long)n_tensors_, (unsigned long long)n_kv_);

        // Read KV metadata (skip values, just parse to advance file position)
        for (uint64_t i = 0; i < n_kv_; i++) {
            std::string key = readString();
            uint32_t vtype;
            file_.read(reinterpret_cast<char*>(&vtype), 4);
            skipValue(vtype);
        }

        // Read tensor infos
        for (uint64_t i = 0; i < n_tensors_; i++) {
            TensorInfo ti;
            ti.name = readString();
            uint32_t n_dims;
            file_.read(reinterpret_cast<char*>(&n_dims), 4);
            ti.n_dims = n_dims;
            ti.total_elements = 1;
            for (int d = 0; d < (int)n_dims; d++) {
                uint64_t dim;
                file_.read(reinterpret_cast<char*>(&dim), 8);
                ti.dims[d] = dim;
                ti.total_elements *= dim;
            }
            for (int d = n_dims; d < 4; d++) ti.dims[d] = 1;

            uint32_t type;
            file_.read(reinterpret_cast<char*>(&type), 4);
            ti.type = type;

            uint64_t offset;
            file_.read(reinterpret_cast<char*>(&offset), 8);
            ti.offset = offset;

            // Calculate data size
            size_t bs = ggml_type_block_size(type);
            size_t bpb = ggml_type_bytes_per_block(type);
            ti.data_size = (ti.total_elements / bs) * bpb;

            tensors_[ti.name] = ti;
        }

        // Record data section offset (aligned to 32 bytes)
        data_offset_ = file_.tellg();
        data_offset_ = (data_offset_ + 31) & ~31ULL;

        printf("[gguf] Data offset: %llu, %zu tensors parsed\n",
               (unsigned long long)data_offset_, tensors_.size());
        return true;
    }

    // Read tensor data from file into a CPU buffer
    bool ReadTensor(const std::string& name, void* buf, size_t buf_size) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            printf("[gguf] Tensor not found: %s\n", name.c_str());
            return false;
        }
        auto& ti = it->second;
        if (ti.data_size > buf_size) {
            printf("[gguf] Buffer too small for %s: need %zu, have %zu\n",
                   name.c_str(), ti.data_size, buf_size);
            return false;
        }
        file_.seekg(static_cast<std::streamoff>(data_offset_ + ti.offset));
        file_.read(reinterpret_cast<char*>(buf), static_cast<std::streamsize>(ti.data_size));
        return true;
    }

    const TensorInfo* GetTensor(const std::string& name) const {
        auto it = tensors_.find(name);
        return (it != tensors_.end()) ? &it->second : nullptr;
    }

    // Upload tensor directly to GPU
    bool UploadTensor(const std::string& name, void** gpu_ptr) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return false;
        auto& ti = it->second;

        // Read to CPU temp buffer
        std::vector<char> cpu_buf(ti.data_size);
        file_.seekg(static_cast<std::streamoff>(data_offset_ + ti.offset));
        file_.read(cpu_buf.data(), static_cast<std::streamsize>(ti.data_size));

        // Upload to GPU
        cudaMalloc(gpu_ptr, ti.data_size);
        cudaMemcpy(*gpu_ptr, cpu_buf.data(), ti.data_size, cudaMemcpyHostToDevice);
        return true;
    }

    size_t TensorCount() const { return tensors_.size(); }

private:
    std::ifstream file_;
    uint32_t version_ = 0;
    uint64_t n_tensors_ = 0;
    uint64_t n_kv_ = 0;
    uint64_t data_offset_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;

    std::string readString() {
        uint64_t len;
        file_.read(reinterpret_cast<char*>(&len), 8);
        std::string s(len, '\0');
        file_.read(&s[0], static_cast<std::streamsize>(len));
        return s;
    }

    void skipValue(uint32_t type) {
        switch (type) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_BOOL:
                file_.seekg(1, std::ios::cur); break;
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
                file_.seekg(2, std::ios::cur); break;
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32:
                file_.seekg(4, std::ios::cur); break;
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64:
                file_.seekg(8, std::ios::cur); break;
            case GGUF_TYPE_STRING:
                readString(); break;
            case GGUF_TYPE_ARRAY: {
                uint32_t atype;
                file_.read(reinterpret_cast<char*>(&atype), 4);
                uint64_t alen;
                file_.read(reinterpret_cast<char*>(&alen), 8);
                for (uint64_t i = 0; i < alen; i++) skipValue(atype);
                break;
            }
        }
    }
};

// ─── CUDAEngine::LoadModel implementation ────────────────────────────────────

bool CUDAEngine::LoadModel(const char* path, const ModelConfig& config) {
    config_ = config;
    printf("[cuda_engine] Loading %s...\n", path);

    GGUFReader gguf;
    if (!gguf.Open(path)) {
        printf("[cuda_engine] Failed to open GGUF\n");
        return false;
    }

    // Allocate workspace
    int n = config_.n_embd;
    int n_kv = config_.n_kv_head * config_.head_dim;
    cudaMalloc(&buf_hidden_, n * sizeof(half));
    cudaMalloc(&buf_residual_, n * sizeof(half));
    cudaMalloc(&buf_qkv_, (n + 2 * n_kv) * sizeof(half));
    cudaMalloc(&buf_attn_out_, n * sizeof(half));
    cudaMalloc(&buf_ffn_, config_.n_ff * 2 * sizeof(half));
    cudaMalloc(&buf_logits_, config_.n_vocab * sizeof(float));

    // Allocate KV cache
    kv_cache_.max_seq = config_.n_ctx;
    kv_cache_.n_layer = config_.n_layer;
    kv_cache_.n_kv_head = config_.n_kv_head;
    kv_cache_.head_dim = config_.head_dim;
    size_t kv_size = (size_t)config_.n_layer * config_.n_kv_head *
                     config_.n_ctx * config_.head_dim * sizeof(half);
    cudaMalloc(&kv_cache_.k, kv_size);
    cudaMalloc(&kv_cache_.v, kv_size);
    cudaMemset(kv_cache_.k, 0, kv_size);
    cudaMemset(kv_cache_.v, 0, kv_size);

    // Allocate + load layer weights
    weights_.n_layer = config_.n_layer;
    weights_.layers = new LayerWeights[config_.n_layer];
    memset(weights_.layers, 0, sizeof(LayerWeights) * config_.n_layer);

    size_t total_loaded = 0;

    // Token embedding
    if (!gguf.UploadTensor("token_embd.weight", (void**)&weights_.tok_embd)) {
        printf("[cuda_engine] WARN: token_embd.weight not found\n");
    } else {
        total_loaded += gguf.GetTensor("token_embd.weight")->data_size;
    }

    // Output norm + output weight
    gguf.UploadTensor("output_norm.weight", (void**)&weights_.output_norm);
    gguf.UploadTensor("output.weight", (void**)&weights_.output);

    // Per-layer weights
    for (int i = 0; i < config_.n_layer; i++) {
        auto& lw = weights_.layers[i];
        char name[128];

        auto load = [&](const char* suffix, void** ptr) {
            snprintf(name, sizeof(name), "blk.%d.%s", i, suffix);
            if (gguf.UploadTensor(name, ptr)) {
                auto* ti = gguf.GetTensor(name);
                if (ti) total_loaded += ti->data_size;
                return true;
            }
            return false;
        };

        load("attn_norm.weight", (void**)&lw.attn_norm);
        load("attn_q.weight", &lw.wq);
        load("attn_k.weight", &lw.wk);
        load("attn_v.weight", &lw.wv);
        load("attn_output.weight", &lw.wo);
        load("attn_q.bias", (void**)&lw.bq);
        load("attn_k.bias", (void**)&lw.bk);
        load("attn_v.bias", (void**)&lw.bv);
        load("ffn_norm.weight", (void**)&lw.ffn_norm);
        load("ffn_gate.weight", &lw.w_gate);
        load("ffn_up.weight", &lw.w_up);
        load("ffn_down.weight", &lw.w_down);
    }

    printf("[cuda_engine] Loaded %.1f MB of weights to GPU\n", total_loaded / 1e6);
    printf("[cuda_engine] KV cache: %.1f MB\n", kv_size * 2 / 1e6);
    return true;
}

} // namespace cuda
} // namespace infergo
