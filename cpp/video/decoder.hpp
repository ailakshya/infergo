// cpp/video/decoder.hpp
// Hardware-accelerated video decoder using FFmpeg + NVDEC (h264_cuvid).
// Falls back to CPU (h264) when GPU decode is unavailable.

#pragma once

#include <cstdint>
#include <string>

// Forward-declare FFmpeg types so callers don't need FFmpeg headers.
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;
struct AVBufferRef;

namespace infergo {

struct FrameInfo {
    int width;
    int height;
    int64_t pts;         // presentation timestamp in microseconds
    int frame_number;
    bool on_gpu;         // true if pixel data was decoded on GPU (NVDEC)
};

class VideoDecoder {
public:
    /// Open a video source.
    /// url can be a file path, RTSP URL, or V4L2 device (/dev/video0).
    /// hw_accel: attempt NVDEC GPU decode first, fall back to CPU on failure.
    explicit VideoDecoder(const std::string& url, bool hw_accel = true);
    ~VideoDecoder();

    // Non-copyable, non-movable (owns FFmpeg contexts).
    VideoDecoder(const VideoDecoder&) = delete;
    VideoDecoder& operator=(const VideoDecoder&) = delete;

    /// Decode the next frame and return RGB24 pixel data.
    /// out_data: set to point at an internal buffer (width * height * 3 bytes).
    ///           Valid until the next call to next_frame() or close().
    /// info:     filled with frame metadata.
    /// Returns false on EOF or error.
    bool next_frame(uint8_t** out_data, FrameInfo* info);

    /// Decode next frame, return raw YUV without RGB conversion.
    /// Skips sws_scale (~5ms at 1440p). For GPU pipeline that does NV12→RGB on GPU.
    /// out_data: Y plane (full frame), UV interleaved below.
    /// out_linesize: bytes per row (may include padding).
    /// Returns false on EOF/error.
    bool next_frame_yuv(uint8_t** out_data, int* out_linesize, int* out_format, FrameInfo* info);

    int width() const;
    int height() const;
    double fps() const;

    /// Set target output size. sws_scale will resize during RGB conversion
    /// at zero extra cost (same FFmpeg call, just different output dimensions).
    /// Call before next_frame(). Set (0,0) to disable resize and use native size.
    void set_output_size(int w, int h);
    int output_width() const { return out_w_ > 0 ? out_w_ : width(); }
    int output_height() const { return out_h_ > 0 ? out_h_ : height(); }

    void close();
    bool is_open() const;
    bool is_hw_accelerated() const;

private:
    bool open_input(const std::string& url);
    bool open_codec(bool hw_accel);
    bool init_hw_device();
    bool decode_packet(AVFrame* out_frame);
    bool convert_to_rgb(AVFrame* src);
    void alloc_rgb_buffer(int w, int h);

    AVFormatContext* fmt_ctx_       = nullptr;
    AVCodecContext*  codec_ctx_     = nullptr;
    AVFrame*         frame_         = nullptr;
    AVFrame*         frame_rgb_     = nullptr;
    AVPacket*        pkt_           = nullptr;
    SwsContext*      sws_ctx_       = nullptr;
    uint8_t*         rgb_buffer_    = nullptr;
    int              rgb_buf_size_  = 0;
    int              video_stream_idx_ = -1;
    int              frame_count_   = 0;
    bool             hw_accel_      = false;
    bool             open_          = false;
    double           fps_           = 0.0;
    int              out_w_         = 0;  // target output width (0 = native)
    int              out_h_         = 0;  // target output height (0 = native)

    // NVDEC hardware context
    AVBufferRef*     hw_device_ctx_ = nullptr;
    AVFrame*         hw_frame_      = nullptr;  // temp frame for GPU->CPU transfer
};

} // namespace infergo
