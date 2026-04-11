// cpp/video/encoder.hpp
// Hardware-accelerated video encoder using FFmpeg + NVENC (h264_nvenc).
// Falls back to libx264 when GPU encoding is unavailable.

#pragma once

#include <cstdint>
#include <string>

struct AVFormatContext;
struct AVCodecContext;
struct AVStream;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace infergo {

class VideoEncoder {
public:
    /// Create an encoder writing to output_path.
    /// codec: "h264_nvenc" (GPU) or "libx264" (CPU).
    /// If the requested codec fails, falls back to libx264.
    VideoEncoder(const std::string& output_path,
                 int width, int height, int fps,
                 const std::string& codec = "h264_nvenc");
    ~VideoEncoder();

    // Non-copyable, non-movable.
    VideoEncoder(const VideoEncoder&) = delete;
    VideoEncoder& operator=(const VideoEncoder&) = delete;

    /// Encode one RGB24 frame (width * height * 3 bytes).
    bool write_frame(const uint8_t* rgb_data);

    void close();
    bool is_open() const;
    bool is_hw_accelerated() const;

private:
    bool open_output(const std::string& path, int width, int height, int fps,
                     const std::string& codec);
    bool init_codec(const std::string& codec_name, int width, int height, int fps);

    AVFormatContext* fmt_ctx_   = nullptr;
    AVCodecContext*  codec_ctx_ = nullptr;
    AVStream*        stream_    = nullptr;
    AVFrame*         frame_     = nullptr;
    AVPacket*        pkt_       = nullptr;
    SwsContext*      sws_ctx_   = nullptr;
    int64_t          frame_count_ = 0;
    bool             hw_accel_  = false;
    bool             open_      = false;
    int              width_     = 0;
    int              height_    = 0;
};

} // namespace infergo
