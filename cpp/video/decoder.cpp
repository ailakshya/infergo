// cpp/video/decoder.cpp
// Hardware-accelerated video decoder: NVDEC (h264_cuvid) with CPU fallback.

#include "decoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace infergo {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

VideoDecoder::VideoDecoder(const std::string& url, bool hw_accel) {
    if (!open_input(url)) {
        return;
    }
    if (!open_codec(hw_accel)) {
        close();
        return;
    }
    frame_     = av_frame_alloc();
    frame_rgb_ = av_frame_alloc();
    hw_frame_  = av_frame_alloc();
    pkt_       = av_packet_alloc();
    if (!frame_ || !frame_rgb_ || !hw_frame_ || !pkt_) {
        fprintf(stderr, "[VideoDecoder] failed to allocate frames/packet\n");
        close();
        return;
    }
    open_ = true;
}

VideoDecoder::~VideoDecoder() {
    close();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool VideoDecoder::next_frame(uint8_t** out_data, FrameInfo* info) {
    if (!open_) return false;

    while (true) {
        int ret = av_read_frame(fmt_ctx_, pkt_);
        if (ret < 0) {
            // EOF or error — try to flush the decoder.
            if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) {
                // Send flush (NULL packet).
                avcodec_send_packet(codec_ctx_, nullptr);
                if (decode_packet(frame_)) {
                    if (!convert_to_rgb(frame_)) return false;
                    if (out_data) *out_data = rgb_buffer_;
                    if (info) {
                        info->width        = codec_ctx_->width;
                        info->height       = codec_ctx_->height;
                        info->pts          = frame_->pts != AV_NOPTS_VALUE
                            ? av_rescale_q(frame_->pts,
                                           fmt_ctx_->streams[video_stream_idx_]->time_base,
                                           AVRational{1, 1000000})
                            : 0;
                        info->frame_number = frame_count_;
                        info->on_gpu       = hw_accel_;
                    }
                    frame_count_++;
                    return true;
                }
            }
            return false;  // real error or end of flush
        }

        if (pkt_->stream_index != video_stream_idx_) {
            av_packet_unref(pkt_);
            continue;
        }

        ret = avcodec_send_packet(codec_ctx_, pkt_);
        av_packet_unref(pkt_);
        if (ret < 0) {
            // NVDEC may fail at runtime (CUDA_ERROR_NOT_SUPPORTED on some GPUs).
            // Auto-fallback to CPU decoder and retry from the start of the file.
            if (hw_accel_) {
                fprintf(stderr, "[VideoDecoder] NVDEC decode failed (%d), falling back to CPU\n", ret);
                // Close current codec and reopen with CPU
                avcodec_free_context(&codec_ctx_);
                if (hw_device_ctx_) { av_buffer_unref(&hw_device_ctx_); hw_device_ctx_ = nullptr; }
                hw_accel_ = false;
                // Reopen codec with CPU
                if (!open_codec(false)) {
                    fprintf(stderr, "[VideoDecoder] CPU fallback also failed\n");
                    return false;
                }
                // Seek back to start
                av_seek_frame(fmt_ctx_, video_stream_idx_, 0, AVSEEK_FLAG_BACKWARD);
                frame_count_ = 0;
                continue;  // retry from beginning with CPU
            }
            fprintf(stderr, "[VideoDecoder] avcodec_send_packet error: %d\n", ret);
            return false;
        }

        if (decode_packet(frame_)) {
            if (!convert_to_rgb(frame_)) return false;
            if (out_data) *out_data = rgb_buffer_;
            if (info) {
                info->width        = codec_ctx_->width;
                info->height       = codec_ctx_->height;
                info->pts          = frame_->pts != AV_NOPTS_VALUE
                    ? av_rescale_q(frame_->pts,
                                   fmt_ctx_->streams[video_stream_idx_]->time_base,
                                   AVRational{1, 1000000})
                    : 0;
                info->frame_number = frame_count_;
                info->on_gpu       = hw_accel_;
            }
            frame_count_++;
            return true;
        }
        // Need more packets for this frame — continue reading.
    }
}

int VideoDecoder::width() const {
    return codec_ctx_ ? codec_ctx_->width : 0;
}

int VideoDecoder::height() const {
    return codec_ctx_ ? codec_ctx_->height : 0;
}

double VideoDecoder::fps() const {
    return fps_;
}

bool VideoDecoder::is_open() const {
    return open_;
}

bool VideoDecoder::is_hw_accelerated() const {
    return hw_accel_;
}

void VideoDecoder::close() {
    open_ = false;
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    if (frame_) {
        av_frame_free(&frame_);
    }
    if (frame_rgb_) {
        av_frame_free(&frame_rgb_);
    }
    if (hw_frame_) {
        av_frame_free(&hw_frame_);
    }
    if (pkt_) {
        av_packet_free(&pkt_);
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
    }
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
    }
    if (rgb_buffer_) {
        av_free(rgb_buffer_);
        rgb_buffer_   = nullptr;
        rgb_buf_size_ = 0;
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

bool VideoDecoder::open_input(const std::string& url) {
    int ret = avformat_open_input(&fmt_ctx_, url.c_str(), nullptr, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[VideoDecoder] cannot open '%s': %d\n", url.c_str(), ret);
        return false;
    }

    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[VideoDecoder] cannot find stream info: %d\n", ret);
        avformat_close_input(&fmt_ctx_);
        return false;
    }

    // Find best video stream.
    video_stream_idx_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO,
                                            -1, -1, nullptr, 0);
    if (video_stream_idx_ < 0) {
        fprintf(stderr, "[VideoDecoder] no video stream found\n");
        avformat_close_input(&fmt_ctx_);
        return false;
    }

    AVStream* vs = fmt_ctx_->streams[video_stream_idx_];
    if (vs->avg_frame_rate.den > 0) {
        fps_ = av_q2d(vs->avg_frame_rate);
    } else if (vs->r_frame_rate.den > 0) {
        fps_ = av_q2d(vs->r_frame_rate);
    }

    return true;
}

bool VideoDecoder::open_codec(bool hw_accel) {
    AVStream* vs = fmt_ctx_->streams[video_stream_idx_];

    // Try GPU codec first if requested.
    const AVCodec* codec = nullptr;
    if (hw_accel) {
        codec = avcodec_find_decoder_by_name("h264_cuvid");
        if (codec) {
            codec_ctx_ = avcodec_alloc_context3(codec);
            if (!codec_ctx_) {
                fprintf(stderr, "[VideoDecoder] failed to alloc codec context\n");
                return false;
            }
            avcodec_parameters_to_context(codec_ctx_, vs->codecpar);

            // h264_cuvid manages its own CUDA context internally.
            // Do NOT set hw_device_ctx — it conflicts with cuvid's internal NVDEC init.
            int ret = avcodec_open2(codec_ctx_, codec, nullptr);
            if (ret >= 0) {
                hw_accel_ = true;
                fprintf(stderr, "[VideoDecoder] NVDEC (h264_cuvid) initialized\n");
                return true;
            }
            fprintf(stderr, "[VideoDecoder] h264_cuvid open failed (%d), falling back to CPU\n", ret);

            // Cleanup failed GPU attempt.
            avcodec_free_context(&codec_ctx_);
        } else {
            fprintf(stderr, "[VideoDecoder] h264_cuvid not found, falling back to CPU\n");
        }
    }

    // CPU fallback.
    codec = avcodec_find_decoder(vs->codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "[VideoDecoder] no CPU decoder found for codec_id %d\n",
                vs->codecpar->codec_id);
        return false;
    }
    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        fprintf(stderr, "[VideoDecoder] failed to alloc codec context (CPU)\n");
        return false;
    }
    avcodec_parameters_to_context(codec_ctx_, vs->codecpar);

    int ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[VideoDecoder] CPU codec open failed: %d\n", ret);
        avcodec_free_context(&codec_ctx_);
        return false;
    }
    hw_accel_ = false;
    fprintf(stderr, "[VideoDecoder] CPU decoder initialized (%s)\n", codec->name);
    return true;
}

bool VideoDecoder::init_hw_device() {
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA,
                                     nullptr, nullptr, 0);
    if (ret < 0) {
        fprintf(stderr, "[VideoDecoder] av_hwdevice_ctx_create(CUDA) failed: %d\n", ret);
        return false;
    }
    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    return true;
}

bool VideoDecoder::decode_packet(AVFrame* out_frame) {
    int ret = avcodec_receive_frame(codec_ctx_, out_frame);
    if (ret == 0) {
        // If frame is on GPU, transfer to CPU.
        if (out_frame->format == AV_PIX_FMT_CUDA) {
            av_frame_unref(hw_frame_);
            ret = av_hwframe_transfer_data(hw_frame_, out_frame, 0);
            if (ret < 0) {
                fprintf(stderr, "[VideoDecoder] GPU->CPU transfer failed: %d\n", ret);
                return false;
            }
            hw_frame_->pts = out_frame->pts;
            // Swap: copy hw_frame_ contents into out_frame for conversion.
            av_frame_unref(out_frame);
            av_frame_move_ref(out_frame, hw_frame_);
        }
        return true;
    }
    if (ret == AVERROR(EAGAIN)) {
        return false;  // need more packets
    }
    if (ret == AVERROR_EOF) {
        return false;
    }
    fprintf(stderr, "[VideoDecoder] decode error: %d\n", ret);
    return false;
}

void VideoDecoder::alloc_rgb_buffer(int w, int h) {
    int needed = av_image_get_buffer_size(AV_PIX_FMT_RGB24, w, h, 1);
    if (needed <= rgb_buf_size_) return;

    if (rgb_buffer_) av_free(rgb_buffer_);
    rgb_buffer_ = static_cast<uint8_t*>(av_malloc(static_cast<size_t>(needed)));
    rgb_buf_size_ = needed;
}

bool VideoDecoder::next_frame_yuv(uint8_t** out_data, int* out_linesize,
                                   int* out_format, FrameInfo* info) {
    if (!open_) return false;

    // Same decode loop as next_frame but skip RGB conversion
    while (true) {
        if (decode_packet(frame_)) {
            if (out_data) *out_data = frame_->data[0];
            if (out_linesize) *out_linesize = frame_->linesize[0];
            if (out_format) *out_format = frame_->format;
            if (info) {
                info->width = codec_ctx_->width;
                info->height = codec_ctx_->height;
                info->pts = frame_->pts;
                info->frame_number = frame_count_;
                info->on_gpu = hw_accel_;
            }
            frame_count_++;
            return true;
        }

        int ret = av_read_frame(fmt_ctx_, pkt_);
        if (ret < 0) return false;
        if (pkt_->stream_index != video_stream_idx_) {
            av_packet_unref(pkt_);
            continue;
        }

        ret = avcodec_send_packet(codec_ctx_, pkt_);
        av_packet_unref(pkt_);
        if (ret < 0) {
            if (hw_accel_) {
                hw_accel_ = false;
                avcodec_free_context(&codec_ctx_);
                if (!open_codec(false)) return false;
                av_seek_frame(fmt_ctx_, video_stream_idx_, 0, AVSEEK_FLAG_BACKWARD);
                frame_count_ = 0;
                continue;
            }
            return false;
        }

        if (decode_packet(frame_)) {
            if (out_data) *out_data = frame_->data[0];
            if (out_linesize) *out_linesize = frame_->linesize[0];
            if (out_format) *out_format = frame_->format;
            if (info) {
                info->width = codec_ctx_->width;
                info->height = codec_ctx_->height;
                info->pts = frame_->pts;
                info->frame_number = frame_count_;
                info->on_gpu = hw_accel_;
            }
            frame_count_++;
            return true;
        }
    }
}

bool VideoDecoder::convert_to_rgb(AVFrame* src) {
    int w = src->width;
    int h = src->height;

    alloc_rgb_buffer(w, h);
    if (!rgb_buffer_) {
        fprintf(stderr, "[VideoDecoder] failed to allocate RGB buffer\n");
        return false;
    }

    // Recreate SwsContext if dimensions or format changed.
    auto src_fmt = static_cast<AVPixelFormat>(src->format);
    sws_ctx_ = sws_getCachedContext(sws_ctx_,
                                    w, h, src_fmt,
                                    w, h, AV_PIX_FMT_RGB24,
                                    SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        fprintf(stderr, "[VideoDecoder] sws_getCachedContext failed\n");
        return false;
    }

    // Set up frame_rgb_ to point at our reusable buffer.
    av_image_fill_arrays(frame_rgb_->data, frame_rgb_->linesize,
                         rgb_buffer_, AV_PIX_FMT_RGB24, w, h, 1);

    sws_scale(sws_ctx_,
              src->data, src->linesize, 0, h,
              frame_rgb_->data, frame_rgb_->linesize);

    return true;
}

} // namespace infergo
