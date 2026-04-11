// cpp/video/frame_annotator.cpp
// Fast frame annotation: draw primitives + TurboJPEG/OpenCV JPEG encoding.

#include "frame_annotator.hpp"
#include "font5x8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef INFER_TURBOJPEG_AVAILABLE
#include <turbojpeg.h>
#endif

#ifdef INFER_OPENCV_AVAILABLE
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace infergo {

// ─── Helpers ─────────────────────────────────────────────────────────────────

static inline int clamp(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline void set_pixel(uint8_t* rgb, int w, int h,
                              int x, int y,
                              uint8_t r, uint8_t g, uint8_t b) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    int off = (y * w + x) * 3;
    rgb[off + 0] = r;
    rgb[off + 1] = g;
    rgb[off + 2] = b;
}

static inline void blend_pixel(uint8_t* rgb, int w, int h,
                                int x, int y,
                                uint8_t r, uint8_t g, uint8_t b, uint8_t alpha) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    int off = (y * w + x) * 3;
    uint16_t a = alpha;
    uint16_t ia = 255 - a;
    rgb[off + 0] = static_cast<uint8_t>((a * r + ia * rgb[off + 0]) / 255);
    rgb[off + 1] = static_cast<uint8_t>((a * g + ia * rgb[off + 1]) / 255);
    rgb[off + 2] = static_cast<uint8_t>((a * b + ia * rgb[off + 2]) / 255);
}

// 20-color class palette matching Go's classPalette.
static const uint8_t CLASS_PALETTE[20][3] = {
    {255,   0,   0},   // red
    {  0, 255,   0},   // green
    {  0,   0, 255},   // blue
    {255, 255,   0},   // yellow
    {255,   0, 255},   // magenta
    {  0, 255, 255},   // cyan
    {255, 128,   0},   // orange
    {128,   0, 255},   // purple
    {  0, 128, 255},   // sky blue
    {255,   0, 128},   // pink
    {128, 255,   0},   // lime
    {  0, 255, 128},   // spring green
    {255, 128, 128},   // salmon
    {128, 128, 255},   // light blue
    {128, 255, 128},   // light green
    {255, 255, 128},   // light yellow
    {255, 128, 255},   // light magenta
    {128, 255, 255},   // light cyan
    {192, 192, 192},   // silver
    {255, 192, 128},   // peach
};

// ─── JPEG encoding helper ────────────────────────────────────────────────────

// Encode an RGB buffer to JPEG. Returns a malloc'd buffer and sets *out_size.
// Returns nullptr on failure.
static uint8_t* encode_jpeg(const uint8_t* rgb, int w, int h,
                            int quality, int* out_size) {
#ifdef INFER_TURBOJPEG_AVAILABLE
    thread_local tjhandle tj = tjInitCompress();
    if (!tj) return nullptr;

    unsigned char* jpeg_buf = nullptr;
    unsigned long jpeg_size = 0;

    int ret = tjCompress2(tj,
                          rgb, w, 0, h, TJPF_BGR,
                          &jpeg_buf, &jpeg_size,
                          TJSAMP_420, quality, TJFLAG_FASTDCT);
    if (ret != 0) {
        if (jpeg_buf) tjFree(jpeg_buf);
        return nullptr;
    }

    // Copy to malloc'd buffer so caller can use free()
    auto* out = static_cast<uint8_t*>(malloc(jpeg_size));
    if (!out) {
        tjFree(jpeg_buf);
        return nullptr;
    }
    memcpy(out, jpeg_buf, jpeg_size);
    tjFree(jpeg_buf);

    *out_size = static_cast<int>(jpeg_size);
    return out;
#else
    // No TurboJPEG available — return error.
    (void)rgb; (void)w; (void)h; (void)quality; (void)out_size;
    return nullptr;
#endif
}

// ─── Nearest-neighbor resize fallback ────────────────────────────────────────

#ifndef INFER_OPENCV_AVAILABLE
static void nn_resize(const uint8_t* src, int sw, int sh,
                      uint8_t* dst, int dw, int dh) {
    for (int y = 0; y < dh; ++y) {
        int sy = y * sh / dh;
        if (sy >= sh) sy = sh - 1;
        for (int x = 0; x < dw; ++x) {
            int sx = x * sw / dw;
            if (sx >= sw) sx = sw - 1;
            int si = (sy * sw + sx) * 3;
            int di = (y * dw + x) * 3;
            dst[di + 0] = src[si + 0];
            dst[di + 1] = src[si + 1];
            dst[di + 2] = src[si + 2];
        }
    }
}
#endif  // !INFER_OPENCV_AVAILABLE

// ─── draw_rect ───────────────────────────────────────────────────────────────

void FrameAnnotator::draw_rect(uint8_t* rgb, int w, int h,
                                int x1, int y1, int x2, int y2,
                                uint8_t r, uint8_t g, uint8_t b, int thickness) {
    if (!rgb || w <= 0 || h <= 0) return;
    x1 = clamp(x1, 0, w - 1);
    y1 = clamp(y1, 0, h - 1);
    x2 = clamp(x2, 0, w - 1);
    y2 = clamp(y2, 0, h - 1);

    for (int t = 0; t < thickness; ++t) {
        // Top edge
        int ty = y1 + t;
        if (ty >= 0 && ty < h) {
            for (int x = x1; x <= x2; ++x)
                set_pixel(rgb, w, h, x, ty, r, g, b);
        }
        // Bottom edge
        int by = y2 - t;
        if (by >= 0 && by < h) {
            for (int x = x1; x <= x2; ++x)
                set_pixel(rgb, w, h, x, by, r, g, b);
        }
        // Left edge
        int lx = x1 + t;
        if (lx >= 0 && lx < w) {
            for (int y = y1; y <= y2; ++y)
                set_pixel(rgb, w, h, lx, y, r, g, b);
        }
        // Right edge
        int rx = x2 - t;
        if (rx >= 0 && rx < w) {
            for (int y = y1; y <= y2; ++y)
                set_pixel(rgb, w, h, rx, y, r, g, b);
        }
    }
}

// ─── draw_text ───────────────────────────────────────────────────────────────

void FrameAnnotator::draw_text(uint8_t* rgb, int w, int h,
                                int x, int y, const char* text,
                                uint8_t r, uint8_t g, uint8_t b, int scale) {
    if (!rgb || !text || w <= 0 || h <= 0 || scale <= 0) return;

    int char_w = 5 * scale;
    int char_h = 8 * scale;
    int len = static_cast<int>(strlen(text));

    // Draw dark background rectangle for readability
    int bg_x1 = x;
    int bg_y1 = y;
    int bg_x2 = x + len * (char_w + scale) - scale; // -scale to remove trailing gap
    int bg_y2 = y + char_h + scale;                  // +scale for padding
    draw_filled_rect(rgb, w, h, bg_x1 - 1, bg_y1 - 1, bg_x2 + 1, bg_y2, 0, 0, 0, 160);

    int cx = x;
    for (int i = 0; i < len; ++i) {
        uint8_t ch = static_cast<uint8_t>(text[i]);
        const uint8_t* glyph;
        uint8_t fallback[8] = {0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F, 0x1F};
        if (ch >= 32 && ch <= 126) {
            glyph = FONT5X8[ch - 32];
        } else {
            glyph = fallback;
        }

        for (int row = 0; row < 8; ++row) {
            uint8_t bits = glyph[row];
            for (int col = 0; col < 5; ++col) {
                if (bits & (1 << (4 - col))) {
                    // Draw a scale x scale block for this pixel
                    for (int sy = 0; sy < scale; ++sy) {
                        for (int sx = 0; sx < scale; ++sx) {
                            int px = cx + col * scale + sx;
                            int py = y + row * scale + sy;
                            set_pixel(rgb, w, h, px, py, r, g, b);
                        }
                    }
                }
            }
        }
        cx += char_w + scale; // char width + inter-char spacing
    }
}

// ─── draw_line ───────────────────────────────────────────────────────────────

void FrameAnnotator::draw_line(uint8_t* rgb, int w, int h,
                                int x1, int y1, int x2, int y2,
                                uint8_t r, uint8_t g, uint8_t b, int thickness) {
    if (!rgb || w <= 0 || h <= 0) return;

    int half = thickness / 2;

    // Bresenham's line algorithm
    int dx = std::abs(x2 - x1);
    int dy = -std::abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx + dy;

    int cx = x1, cy = y1;
    for (;;) {
        // Draw a thickness x thickness block centered on (cx, cy)
        for (int ty = cy - half; ty <= cy + half; ++ty) {
            for (int tx = cx - half; tx <= cx + half; ++tx) {
                set_pixel(rgb, w, h, tx, ty, r, g, b);
            }
        }

        if (cx == x2 && cy == y2) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; cx += sx; }
        if (e2 <= dx) { err += dx; cy += sy; }
    }
}

// ─── draw_polygon ────────────────────────────────────────────────────────────

void FrameAnnotator::draw_polygon(uint8_t* rgb, int w, int h,
                                   const Point* pts, int n,
                                   uint8_t r, uint8_t g, uint8_t b, uint8_t alpha) {
    if (!rgb || !pts || n < 3 || w <= 0 || h <= 0) return;

    // 1) Alpha-blended fill using scanline ray casting
    int min_x = pts[0].x, max_x = pts[0].x;
    int min_y = pts[0].y, max_y = pts[0].y;
    for (int i = 1; i < n; ++i) {
        if (pts[i].x < min_x) min_x = pts[i].x;
        if (pts[i].x > max_x) max_x = pts[i].x;
        if (pts[i].y < min_y) min_y = pts[i].y;
        if (pts[i].y > max_y) max_y = pts[i].y;
    }
    min_x = clamp(min_x, 0, w - 1);
    max_x = clamp(max_x, 0, w - 1);
    min_y = clamp(min_y, 0, h - 1);
    max_y = clamp(max_y, 0, h - 1);

    if (alpha > 0) {
        for (int py = min_y; py <= max_y; ++py) {
            for (int px = min_x; px <= max_x; ++px) {
                // Ray casting: count crossings to the right of (px, py)
                int crossings = 0;
                for (int i = 0; i < n; ++i) {
                    int j = (i + 1) % n;
                    int yi = pts[i].y, yj = pts[j].y;
                    int xi = pts[i].x, xj = pts[j].x;
                    if ((yi <= py && yj > py) || (yj <= py && yi > py)) {
                        float vt = static_cast<float>(py - yi) / static_cast<float>(yj - yi);
                        float ix = static_cast<float>(xi) + vt * static_cast<float>(xj - xi);
                        if (static_cast<float>(px) < ix) ++crossings;
                    }
                }
                if (crossings & 1) {
                    blend_pixel(rgb, w, h, px, py, r, g, b, alpha);
                }
            }
        }
    }

    // 2) Draw outline by connecting consecutive vertices
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        draw_line(rgb, w, h, pts[i].x, pts[i].y, pts[j].x, pts[j].y, r, g, b, 2);
    }
}

// ─── draw_filled_rect ────────────────────────────────────────────────────────

void FrameAnnotator::draw_filled_rect(uint8_t* rgb, int w, int h,
                                       int x1, int y1, int x2, int y2,
                                       uint8_t r, uint8_t g, uint8_t b, uint8_t alpha) {
    if (!rgb || w <= 0 || h <= 0) return;
    x1 = clamp(x1, 0, w - 1);
    y1 = clamp(y1, 0, h - 1);
    x2 = clamp(x2, 0, w - 1);
    y2 = clamp(y2, 0, h - 1);

    for (int y = y1; y <= y2; ++y) {
        for (int x = x1; x <= x2; ++x) {
            blend_pixel(rgb, w, h, x, y, r, g, b, alpha);
        }
    }
}

// ─── annotate_jpeg ───────────────────────────────────────────────────────────

uint8_t* FrameAnnotator::annotate_jpeg(const uint8_t* rgb, int w, int h,
                                        const AnnotateBox* boxes, int n,
                                        int quality, int* out_size) {
    if (!rgb || w <= 0 || h <= 0 || !out_size) return nullptr;
    if (quality <= 0) quality = 75;

    // Copy RGB to working buffer
    size_t buf_size = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
    std::vector<uint8_t> buf(buf_size);
    memcpy(buf.data(), rgb, buf_size);

    // Draw each box
    for (int i = 0; i < n; ++i) {
        const AnnotateBox& box = boxes[i];
        int ci = ((box.class_id % 20) + 20) % 20;
        uint8_t cr = CLASS_PALETTE[ci][0];
        uint8_t cg = CLASS_PALETTE[ci][1];
        uint8_t cb = CLASS_PALETTE[ci][2];

        int bx1 = clamp(static_cast<int>(box.x1), 0, w - 1);
        int by1 = clamp(static_cast<int>(box.y1), 0, h - 1);
        int bx2 = clamp(static_cast<int>(box.x2), 0, w - 1);
        int by2 = clamp(static_cast<int>(box.y2), 0, h - 1);

        // Draw bounding box (2px thick)
        draw_rect(buf.data(), w, h, bx1, by1, bx2, by2, cr, cg, cb, 2);

        // Build label string
        char label[128];
        if (box.label[0] != '\0') {
            snprintf(label, sizeof(label), "%s", box.label);
        } else {
            snprintf(label, sizeof(label), "ID:%d cls%d: %.0f%%",
                     box.track_id, box.class_id, static_cast<double>(box.confidence) * 100.0);
        }

        // Draw label above box
        int text_y = by1 - 10;
        if (text_y < 0) text_y = by1 + 2;
        draw_text(buf.data(), w, h, bx1, text_y, label, 255, 255, 255, 1);
    }

    return encode_jpeg(buf.data(), w, h, quality, out_size);
}

// ─── resize_jpeg ─────────────────────────────────────────────────────────────

uint8_t* FrameAnnotator::resize_jpeg(const uint8_t* rgb, int sw, int sh,
                                      int dw, int dh,
                                      int quality, int* out_size) {
    if (!rgb || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || !out_size) return nullptr;
    if (quality <= 0) quality = 75;

#ifdef INFER_OPENCV_AVAILABLE
    // Use OpenCV for high-quality bilinear resize
    cv::Mat src(sh, sw, CV_8UC3, const_cast<uint8_t*>(rgb));
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(dw, dh), 0, 0, cv::INTER_LINEAR);
    return encode_jpeg(dst.data, dw, dh, quality, out_size);
#else
    // Fallback: nearest-neighbor resize
    std::vector<uint8_t> dst(static_cast<size_t>(dw) * static_cast<size_t>(dh) * 3);
    nn_resize(rgb, sw, sh, dst.data(), dw, dh);
    return encode_jpeg(dst.data(), dw, dh, quality, out_size);
#endif
}

// ─── combine_jpeg ────────────────────────────────────────────────────────────

uint8_t* FrameAnnotator::combine_jpeg(const uint8_t* rgb1, int w1, int h1,
                                       const uint8_t* rgb2, int w2, int h2,
                                       const char* status,
                                       int tw, int th,
                                       int quality, int* out_size) {
    if (!rgb1 || !rgb2 || tw <= 0 || th <= 0 || !out_size) return nullptr;
    if (quality <= 0) quality = 75;

    int status_bar_h = 40;
    int frame_h = th - status_bar_h;
    if (frame_h <= 0) frame_h = th;
    int half_w = tw / 2;

    // Resize both frames to (half_w, frame_h)
    std::vector<uint8_t> left(static_cast<size_t>(half_w) * static_cast<size_t>(frame_h) * 3);
    std::vector<uint8_t> right(static_cast<size_t>(half_w) * static_cast<size_t>(frame_h) * 3);

#ifdef INFER_OPENCV_AVAILABLE
    {
        cv::Mat src1(h1, w1, CV_8UC3, const_cast<uint8_t*>(rgb1));
        cv::Mat dst1;
        cv::resize(src1, dst1, cv::Size(half_w, frame_h), 0, 0, cv::INTER_LINEAR);
        memcpy(left.data(), dst1.data, left.size());
    }
    {
        cv::Mat src2(h2, w2, CV_8UC3, const_cast<uint8_t*>(rgb2));
        cv::Mat dst2;
        cv::resize(src2, dst2, cv::Size(half_w, frame_h), 0, 0, cv::INTER_LINEAR);
        memcpy(right.data(), dst2.data, right.size());
    }
#else
    nn_resize(rgb1, w1, h1, left.data(), half_w, frame_h);
    nn_resize(rgb2, w2, h2, right.data(), half_w, frame_h);
#endif

    // Allocate combined canvas (black)
    size_t canvas_size = static_cast<size_t>(tw) * static_cast<size_t>(th) * 3;
    std::vector<uint8_t> canvas(canvas_size, 0);

    // Copy left frame
    for (int y = 0; y < frame_h; ++y) {
        memcpy(&canvas[static_cast<size_t>(y * tw) * 3],
               &left[static_cast<size_t>(y * half_w) * 3],
               static_cast<size_t>(half_w) * 3);
    }

    // Copy right frame
    for (int y = 0; y < frame_h; ++y) {
        memcpy(&canvas[(static_cast<size_t>(y * tw) + half_w) * 3],
               &right[static_cast<size_t>(y * half_w) * 3],
               static_cast<size_t>(half_w) * 3);
    }

    // Draw status bar background
    draw_filled_rect(canvas.data(), tw, th,
                     0, frame_h, tw - 1, th - 1,
                     40, 40, 40, 255);

    // Draw status text centered in the bar
    if (status && status[0] != '\0') {
        int text_scale = 2;
        int char_w = 5 * text_scale + text_scale; // char width + spacing
        int text_len = static_cast<int>(strlen(status));
        int text_w = text_len * char_w - text_scale;
        int text_x = (tw - text_w) / 2;
        if (text_x < 4) text_x = 4;
        int text_y = frame_h + (status_bar_h - 8 * text_scale) / 2;
        draw_text(canvas.data(), tw, th, text_x, text_y, status, 255, 255, 255, text_scale);
    }

    return encode_jpeg(canvas.data(), tw, th, quality, out_size);
}

// ─── jpeg_free ───────────────────────────────────────────────────────────────

void FrameAnnotator::jpeg_free(uint8_t* buf) {
    free(buf);
}

uint8_t* FrameAnnotator::annotate_full(
    const uint8_t* rgb, int w, int h,
    const AnnotateBox* boxes, int n_boxes,
    const Line* lines, int n_lines,
    const PolygonOverlay* polygons, int n_polygons,
    const TextOverlay* texts, int n_texts,
    const FilledRect* rects, int n_rects,
    int out_w, int out_h, int quality, int* out_size)
{
    if (!rgb || w <= 0 || h <= 0 || !out_size) return nullptr;

    // 1. Copy input to working buffer.
    size_t nbytes = static_cast<size_t>(w) * h * 3;
    auto* buf = static_cast<uint8_t*>(malloc(nbytes));
    if (!buf) return nullptr;
    memcpy(buf, rgb, nbytes);

    // 2. Draw filled rects (backgrounds — draw first so text goes on top).
    for (int i = 0; i < n_rects; i++) {
        draw_filled_rect(buf, w, h,
            rects[i].x1, rects[i].y1, rects[i].x2, rects[i].y2,
            rects[i].r, rects[i].g, rects[i].b, rects[i].alpha);
    }

    // 3. Draw polygons (zone overlays).
    for (int i = 0; i < n_polygons; i++) {
        draw_polygon(buf, w, h, polygons[i].pts, polygons[i].n_pts,
            polygons[i].r, polygons[i].g, polygons[i].b, polygons[i].alpha);
    }

    // 4. Draw lines (counting lines).
    for (int i = 0; i < n_lines; i++) {
        draw_line(buf, w, h, lines[i].x1, lines[i].y1, lines[i].x2, lines[i].y2,
            lines[i].r, lines[i].g, lines[i].b, lines[i].thickness);
    }

    // 5. Draw bounding boxes with labels.
    static const uint8_t palette[20][3] = {
        {255,0,0},{0,255,0},{0,0,255},{255,255,0},{255,0,255},
        {0,255,255},{255,128,0},{128,0,255},{0,128,255},{255,0,128},
        {128,255,0},{0,255,128},{255,128,128},{128,128,255},{128,255,128},
        {255,255,128},{255,128,255},{128,255,255},{192,192,192},{255,192,128}
    };
    for (int i = 0; i < n_boxes; i++) {
        int ci = boxes[i].class_id % 20;
        uint8_t cr = palette[ci][0], cg = palette[ci][1], cb = palette[ci][2];
        int bx1 = std::max(0, static_cast<int>(boxes[i].x1));
        int by1 = std::max(0, static_cast<int>(boxes[i].y1));
        int bx2 = std::min(w-1, static_cast<int>(boxes[i].x2));
        int by2 = std::min(h-1, static_cast<int>(boxes[i].y2));
        draw_rect(buf, w, h, bx1, by1, bx2, by2, cr, cg, cb, 2);
        if (boxes[i].label[0] != '\0') {
            int scale = (w >= 2560) ? 3 : (w >= 1920) ? 2 : 1;
            int lw = static_cast<int>(strlen(boxes[i].label)) * 6 * scale;
            int lh = 10 * scale;
            draw_filled_rect(buf, w, h, bx1, by1-lh-2, bx1+lw+4, by1, 0, 0, 0, 180);
            draw_text(buf, w, h, bx1+2, by1-lh, boxes[i].label, 255, 255, 255, scale);
        }
    }

    // 6. Draw text overlays (stats, camera name, etc.).
    for (int i = 0; i < n_texts; i++) {
        draw_text(buf, w, h, texts[i].x, texts[i].y, texts[i].text,
            texts[i].r, texts[i].g, texts[i].b, texts[i].scale);
    }

    // 7. Resize if needed, then JPEG encode.
    uint8_t* result = nullptr;
    if (out_w > 0 && out_h > 0 && (out_w != w || out_h != h)) {
        result = resize_jpeg(buf, w, h, out_w, out_h, quality, out_size);
    } else {
        // Encode at original size.
#ifdef INFER_TURBOJPEG_AVAILABLE
        thread_local tjhandle tj = tjInitCompress();
        unsigned char* jpegBuf = nullptr;
        unsigned long jpegSize = 0;
        int rc = tjCompress2(tj, buf, w, 0, h, TJPF_BGR,
                             &jpegBuf, &jpegSize, TJSAMP_420, quality, TJFLAG_FASTDCT);
        if (rc == 0 && jpegBuf) {
            result = static_cast<uint8_t*>(malloc(jpegSize));
            if (result) {
                memcpy(result, jpegBuf, jpegSize);
                *out_size = static_cast<int>(jpegSize);
            }
            tjFree(jpegBuf);
        }
#endif
    }

    free(buf);
    return result;
}

} // namespace infergo
