// cpp/video/encoder.cpp
// Hardware-accelerated video encoder: NVENC (h264_nvenc) with libx264 fallback.

#include "encoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <cstdio>

namespace infergo {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

VideoEncoder::VideoEncoder(const std::string& output_path,
                           int width, int height, int fps,
                           const std::string& codec) {
    if (!open_output(output_path, width, height, fps, codec)) {
        close();
        return;
    }
    open_ = true;
}

VideoEncoder::~VideoEncoder() {
    close();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool VideoEncoder::write_frame(const uint8_t* rgb_data) {
    if (!open_ || !rgb_data) return false;

    // Convert RGB24 -> encoder pixel format (YUV420P).
    const uint8_t* src_data[1] = { rgb_data };
    int src_linesize[1] = { width_ * 3 };

    sws_scale(sws_ctx_,
              src_data, src_linesize, 0, height_,
              frame_->data, frame_->linesize);

    frame_->pts = frame_count_;
    frame_count_++;

    // Send frame to encoder.
    int ret = avcodec_send_frame(codec_ctx_, frame_);
    if (ret < 0) {
        fprintf(stderr, "[VideoEncoder] avcodec_send_frame error: %d\n", ret);
        return false;
    }

    // Receive and write all available packets.
    while (true) {
        ret = avcodec_receive_packet(codec_ctx_, pkt_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        if (ret < 0) {
            fprintf(stderr, "[VideoEncoder] avcodec_receive_packet error: %d\n", ret);
            return false;
        }
        av_packet_rescale_ts(pkt_, codec_ctx_->time_base, stream_->time_base);
        pkt_->stream_index = stream_->index;

        ret = av_interleaved_write_frame(fmt_ctx_, pkt_);
        av_packet_unref(pkt_);
        if (ret < 0) {
            fprintf(stderr, "[VideoEncoder] write frame error: %d\n", ret);
            return false;
        }
    }
    return true;
}

bool VideoEncoder::is_open() const {
    return open_;
}

bool VideoEncoder::is_hw_accelerated() const {
    return hw_accel_;
}

void VideoEncoder::close() {
    if (!open_ && !fmt_ctx_) return;

    if (open_ && codec_ctx_ && fmt_ctx_) {
        // Flush encoder.
        avcodec_send_frame(codec_ctx_, nullptr);
        while (true) {
            int ret = avcodec_receive_packet(codec_ctx_, pkt_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) break;
            av_packet_rescale_ts(pkt_, codec_ctx_->time_base, stream_->time_base);
            pkt_->stream_index = stream_->index;
            av_interleaved_write_frame(fmt_ctx_, pkt_);
            av_packet_unref(pkt_);
        }
        av_write_trailer(fmt_ctx_);
    }
    open_ = false;

    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    if (frame_) {
        av_frame_free(&frame_);
    }
    if (pkt_) {
        av_packet_free(&pkt_);
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    if (fmt_ctx_) {
        if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&fmt_ctx_->pb);
        }
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

bool VideoEncoder::init_codec(const std::string& codec_name,
                              int width, int height, int fps) {
    const AVCodec* codec = avcodec_find_encoder_by_name(codec_name.c_str());
    if (!codec) {
        fprintf(stderr, "[VideoEncoder] codec '%s' not found\n", codec_name.c_str());
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) return false;

    codec_ctx_->width     = width;
    codec_ctx_->height    = height;
    codec_ctx_->time_base = {1, fps};
    codec_ctx_->framerate = {fps, 1};
    codec_ctx_->pix_fmt   = AV_PIX_FMT_YUV420P;
    codec_ctx_->gop_size  = 12;

    // NVENC: use balanced preset.
    if (codec_name == "h264_nvenc") {
        av_opt_set(codec_ctx_->priv_data, "preset", "p4", 0);
        av_opt_set(codec_ctx_->priv_data, "tune",   "hq", 0);
    }
    // libx264: use medium preset.
    if (codec_name == "libx264") {
        av_opt_set(codec_ctx_->priv_data, "preset", "medium", 0);
    }

    // Set bit rate for reasonable quality.
    codec_ctx_->bit_rate = 4000000;  // 4 Mbps

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    int ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[VideoEncoder] avcodec_open2('%s') failed: %d\n",
                codec_name.c_str(), ret);
        avcodec_free_context(&codec_ctx_);
        return false;
    }

    hw_accel_ = (codec_name == "h264_nvenc");
    return true;
}

bool VideoEncoder::open_output(const std::string& path,
                               int width, int height, int fps,
                               const std::string& codec) {
    width_  = width;
    height_ = height;

    int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, nullptr, path.c_str());
    if (ret < 0 || !fmt_ctx_) {
        fprintf(stderr, "[VideoEncoder] cannot create output context for '%s'\n", path.c_str());
        return false;
    }

    stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!stream_) {
        fprintf(stderr, "[VideoEncoder] cannot create output stream\n");
        return false;
    }

    // Try requested codec, fall back to libx264.
    if (!init_codec(codec, width, height, fps)) {
        if (codec != "libx264") {
            fprintf(stderr, "[VideoEncoder] falling back to libx264\n");
            if (!init_codec("libx264", width, height, fps)) {
                return false;
            }
        } else {
            return false;
        }
    }

    ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
    if (ret < 0) {
        fprintf(stderr, "[VideoEncoder] avcodec_parameters_from_context failed\n");
        return false;
    }
    stream_->time_base = codec_ctx_->time_base;

    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&fmt_ctx_->pb, path.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            fprintf(stderr, "[VideoEncoder] cannot open output file '%s'\n", path.c_str());
            return false;
        }
    }

    ret = avformat_write_header(fmt_ctx_, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[VideoEncoder] avformat_write_header failed: %d\n", ret);
        return false;
    }

    // Allocate frame for encoder input (YUV420P).
    frame_ = av_frame_alloc();
    if (!frame_) return false;
    frame_->format = codec_ctx_->pix_fmt;
    frame_->width  = width;
    frame_->height = height;
    ret = av_frame_get_buffer(frame_, 0);
    if (ret < 0) {
        fprintf(stderr, "[VideoEncoder] av_frame_get_buffer failed\n");
        return false;
    }

    pkt_ = av_packet_alloc();
    if (!pkt_) return false;

    // Create RGB24 -> YUV420P converter.
    sws_ctx_ = sws_getContext(width, height, AV_PIX_FMT_RGB24,
                              width, height, AV_PIX_FMT_YUV420P,
                              SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        fprintf(stderr, "[VideoEncoder] sws_getContext failed\n");
        return false;
    }

    fprintf(stderr, "[VideoEncoder] opened '%s' %dx%d @ %d fps, codec=%s (hw=%s)\n",
            path.c_str(), width, height, fps,
            codec_ctx_->codec->name,
            hw_accel_ ? "NVENC" : "CPU");
    return true;
}

} // namespace infergo
