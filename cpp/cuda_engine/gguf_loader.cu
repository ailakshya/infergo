// GGUF model loader — parses GGUF v3 format and uploads Q4_K weights to GPU
#include "engine.cuh"
#include "q4k_fast.cuh"
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

    // Upload tensor directly to GPU (raw — for Q4_K weight tensors)
    bool UploadTensor(const std::string& name, void** gpu_ptr) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return false;
        auto& ti = it->second;

        std::vector<char> cpu_buf(ti.data_size);
        file_.seekg(static_cast<std::streamoff>(data_offset_ + ti.offset));
        file_.read(cpu_buf.data(), static_cast<std::streamsize>(ti.data_size));

        cudaMalloc(gpu_ptr, ti.data_size);
        cudaMemcpy(*gpu_ptr, cpu_buf.data(), ti.data_size, cudaMemcpyHostToDevice);
        return true;
    }

    // Upload tensor as FP16 — converts F32→F16 or dequantizes Q4_K→F16
    bool UploadTensorAsF16(const std::string& name, half** gpu_ptr) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return false;
        auto& ti = it->second;

        std::vector<char> cpu_buf(ti.data_size);
        file_.seekg(static_cast<std::streamoff>(data_offset_ + ti.offset));
        file_.read(cpu_buf.data(), static_cast<std::streamsize>(ti.data_size));

        size_t n_elements = ti.total_elements;

        if (ti.type == GGML_TYPE_F32) {
            // F32 → F16 conversion
            std::vector<half> f16_buf(n_elements);
            const float* f32 = reinterpret_cast<const float*>(cpu_buf.data());
            for (size_t i = 0; i < n_elements; i++) {
                f16_buf[i] = __float2half(f32[i]);
            }
            cudaMalloc(gpu_ptr, n_elements * sizeof(half));
            cudaMemcpy(*gpu_ptr, f16_buf.data(), n_elements * sizeof(half), cudaMemcpyHostToDevice);
        } else if (ti.type == GGML_TYPE_F16) {
            // Already F16 — direct upload
            cudaMalloc(gpu_ptr, n_elements * sizeof(half));
            cudaMemcpy(*gpu_ptr, cpu_buf.data(), n_elements * sizeof(half), cudaMemcpyHostToDevice);
        } else if (ti.type == GGML_TYPE_Q4_K) {
            // Q4_K → F16 dequantization
            std::vector<half> f16_buf(n_elements);
            size_t n_blocks = n_elements / 256;
            const unsigned char* raw = reinterpret_cast<const unsigned char*>(cpu_buf.data());

            for (size_t b = 0; b < n_blocks; b++) {
                // Parse Q4_K block (144 bytes per block)
                const unsigned char* blk = raw + b * 144;
                // d and dmin are FP16 (2 bytes each)
                half d_h, dmin_h;
                memcpy(&d_h, blk, 2);
                memcpy(&dmin_h, blk + 2, 2);
                float d = __half2float(d_h);
                float dmin = __half2float(dmin_h);
                const unsigned char* scales = blk + 4;    // 12 bytes
                const unsigned char* qs = blk + 16;       // 128 bytes

                for (int j = 0; j < 256; j++) {
                    int sub = j / 32;
                    uint8_t sc, m;
                    if (sub < 4) {
                        sc = scales[sub] & 0x3F;
                        m = scales[sub + 4] & 0x3F;
                    } else {
                        sc = ((scales[sub + 4] & 0xF) | ((scales[sub - 4] >> 6) << 4));
                        m = ((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
                    }
                    float scale = d * sc;
                    float min_val = dmin * m;

                    int byte_idx;
                    if (sub < 4) byte_idx = sub * 16 + (j % 32) / 2;
                    else byte_idx = 64 + (sub - 4) * 16 + (j % 32) / 2;

                    uint8_t byte = qs[byte_idx];
                    int nibble = (j & 1) ? (byte >> 4) : (byte & 0xF);
                    float val = scale * nibble - min_val;

                    f16_buf[b * 256 + j] = __float2half(val);
                }
            }
            cudaMalloc(gpu_ptr, n_elements * sizeof(half));
            cudaMemcpy(*gpu_ptr, f16_buf.data(), n_elements * sizeof(half), cudaMemcpyHostToDevice);
            printf("[gguf] Dequantized %s Q4_K→F16 (%zu elements)\n", name.c_str(), n_elements);
        } else if (ti.type == GGML_TYPE_Q6_K) {
            // Q6_K → F16 dequantization
            // Block: 210 bytes, 256 values
            // ql[128] (4-bit low), qh[64] (2-bit high), scales[16], d(fp16)
            std::vector<half> f16_buf(n_elements);
            size_t n_blocks = n_elements / 256;
            const unsigned char* raw = reinterpret_cast<const unsigned char*>(cpu_buf.data());

            for (size_t b = 0; b < n_blocks; b++) {
                const unsigned char* blk = raw + b * 210;
                const uint8_t* ql = blk;           // 128 bytes
                const uint8_t* qh = blk + 128;     // 64 bytes
                const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192); // 16 bytes
                half d_h;
                memcpy(&d_h, blk + 208, 2);
                float d = __half2float(d_h);

                for (int j = 0; j < 256; j++) {
                    // Low 4 bits from ql
                    int ql_idx = j / 2;
                    int q_low = (j & 1) ? (ql[ql_idx] >> 4) : (ql[ql_idx] & 0xF);

                    // High 2 bits from qh
                    int qh_idx = j / 4;
                    int qh_shift = (j % 4) * 2;
                    int q_high = (qh[qh_idx] >> qh_shift) & 0x3;

                    int q = q_low | (q_high << 4);  // 6-bit value [0..63]
                    q -= 32;  // center around 0 → [-32..31]

                    int scale_idx = j / 16;
                    float val = d * sc[scale_idx] * q;

                    f16_buf[b * 256 + j] = __float2half(val);
                }
            }
            cudaMalloc(gpu_ptr, n_elements * sizeof(half));
            cudaMemcpy(*gpu_ptr, f16_buf.data(), n_elements * sizeof(half), cudaMemcpyHostToDevice);
            printf("[gguf] Dequantized %s Q6_K→F16 (%zu elements)\n", name.c_str(), n_elements);
        } else {
            printf("[gguf] Unsupported type %d for F16 upload: %s\n", ti.type, name.c_str());
            return false;
        }
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

    // Q8_1 buffers for dp4a GEMV
    int q8_blocks = config_.n_embd / 32;
    cudaMalloc(&buf_q8_, q8_blocks * sizeof(BlockQ8_1));
    cudaMalloc(&buf_rms_, sizeof(float));

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

    // Token embedding — dequantize to F16 for direct lookup
    if (!gguf.UploadTensorAsF16("token_embd.weight", &weights_.tok_embd)) {
        printf("[cuda_engine] WARN: token_embd.weight not found\n");
    } else {
        auto* ti = gguf.GetTensor("token_embd.weight");
        if (ti) total_loaded += ti->data_size;
    }

    // Output norm (F32→F16)
    gguf.UploadTensorAsF16("output_norm.weight", &weights_.output_norm);

    // Output weight (Q6_K → F16)
    {
        auto* ti = gguf.GetTensor("output.weight");
        if (ti && ti->type == GGML_TYPE_Q4_K) {
            gguf.UploadTensor("output.weight", (void**)&weights_.output);
            weights_.output_f16 = false;
        } else {
            half* f16_ptr = nullptr;
            gguf.UploadTensorAsF16("output.weight", &f16_ptr);
            weights_.output = f16_ptr;
            weights_.output_f16 = true;
        }
    }

    // Per-layer weights
    for (int i = 0; i < config_.n_layer; i++) {
        auto& lw = weights_.layers[i];
        char name[128];

        // Q4_K → keep quantized (small memory, needs fast kernel)
        // Q6_K/F32 → dequant to F16 (correct output)
        auto load_weight = [&](const char* suffix, void** ptr, bool* is_f16) {
            snprintf(name, sizeof(name), "blk.%d.%s", i, suffix);
            auto* ti = gguf.GetTensor(name);
            if (!ti) return false;

            if (ti->type == GGML_TYPE_Q4_K) {
                if (gguf.UploadTensor(name, ptr)) {
                    total_loaded += ti->data_size;
                    *is_f16 = false;
                    return true;
                }
            } else {
                half* f16_ptr = nullptr;
                if (gguf.UploadTensorAsF16(name, &f16_ptr)) {
                    *ptr = f16_ptr;
                    total_loaded += ti->data_size;
                    *is_f16 = true;
                    return true;
                }
            }
            return false;
        };

        // F32 weights → convert to F16
        auto load_f16 = [&](const char* suffix, half** ptr) {
            snprintf(name, sizeof(name), "blk.%d.%s", i, suffix);
            if (gguf.UploadTensorAsF16(name, ptr)) {
                auto* ti = gguf.GetTensor(name);
                if (ti) total_loaded += ti->data_size;
                return true;
            }
            return false;
        };

        // Norms (F32→F16)
        load_f16("attn_norm.weight", &lw.attn_norm);
        load_f16("ffn_norm.weight", &lw.ffn_norm);

        // Biases (F32→F16)
        load_f16("attn_q.bias", &lw.bq);
        load_f16("attn_k.bias", &lw.bk);
        load_f16("attn_v.bias", &lw.bv);

        // Weight matrices — Q4_K stays quantized, Q6_K → F16
        load_weight("attn_q.weight", &lw.wq, &lw.wq_f16);
        load_weight("attn_k.weight", &lw.wk, &lw.wk_f16);
        load_weight("attn_v.weight", &lw.wv, &lw.wv_f16);
        load_weight("attn_output.weight", &lw.wo, &lw.wo_f16);
        load_weight("ffn_gate.weight", &lw.w_gate, &lw.w_gate_f16);
        load_weight("ffn_up.weight", &lw.w_up, &lw.w_up_f16);
        load_weight("ffn_down.weight", &lw.w_down, &lw.w_down_f16);
    }

    printf("[cuda_engine] Loaded %.1f MB of weights to GPU\n", total_loaded / 1e6);
    printf("[cuda_engine] KV cache: %.1f MB\n", kv_size * 2 / 1e6);
    return true;
}

} // namespace cuda
} // namespace infergo
