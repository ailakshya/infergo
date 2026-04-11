// cpp/video/frame_annotator.hpp
// Fast frame annotation using TurboJPEG + OpenCV.
// Replaces Go's pure-Go image processing for annotated JPEG output.

#pragma once

#include <cstdint>

namespace infergo {

struct AnnotateBox {
    float x1, y1, x2, y2;
    int   class_id;
    float confidence;
    int   track_id;
    char  label[64];
};

struct Point {
    int x, y;
};

class FrameAnnotator {
public:
    /// Draw a rectangle outline on an RGB buffer.
    static void draw_rect(uint8_t* rgb, int w, int h,
                          int x1, int y1, int x2, int y2,
                          uint8_t r, uint8_t g, uint8_t b, int thickness);

    /// Draw text using the built-in 5x8 bitmap font.
    /// scale=1 -> 5x8 px chars, scale=2 -> 10x16 px, etc.
    static void draw_text(uint8_t* rgb, int w, int h,
                          int x, int y, const char* text,
                          uint8_t r, uint8_t g, uint8_t b, int scale);

    /// Draw a line between two points using Bresenham with thickness.
    static void draw_line(uint8_t* rgb, int w, int h,
                          int x1, int y1, int x2, int y2,
                          uint8_t r, uint8_t g, uint8_t b, int thickness);

    /// Draw a polygon outline + alpha-blended fill.
    static void draw_polygon(uint8_t* rgb, int w, int h,
                             const Point* pts, int n,
                             uint8_t r, uint8_t g, uint8_t b, uint8_t alpha);

    /// Draw a filled rectangle with alpha blending.
    static void draw_filled_rect(uint8_t* rgb, int w, int h,
                                 int x1, int y1, int x2, int y2,
                                 uint8_t r, uint8_t g, uint8_t b, uint8_t alpha);

    /// Annotate an RGB frame with bounding boxes and labels, encode as JPEG.
    /// Returns malloc'd JPEG buffer (caller must free with jpeg_free).
    /// Sets *out_size to the JPEG byte count.
    static uint8_t* annotate_jpeg(const uint8_t* rgb, int w, int h,
                                  const AnnotateBox* boxes, int n,
                                  int quality, int* out_size);

    /// Resize an RGB frame and encode as JPEG.
    /// Returns malloc'd JPEG buffer (caller must free with jpeg_free).
    static uint8_t* resize_jpeg(const uint8_t* rgb, int sw, int sh,
                                int dw, int dh,
                                int quality, int* out_size);

    /// Combine two RGB frames side-by-side with a status bar, encode as JPEG.
    /// Returns malloc'd JPEG buffer (caller must free with jpeg_free).
    static uint8_t* combine_jpeg(const uint8_t* rgb1, int w1, int h1,
                                 const uint8_t* rgb2, int w2, int h2,
                                 const char* status,
                                 int tw, int th,
                                 int quality, int* out_size);

    /// Free a buffer returned by annotate_jpeg / resize_jpeg / combine_jpeg.
    static void jpeg_free(uint8_t* buf);

    /// Batch annotate: draw ALL overlays on a frame, resize, and JPEG encode
    /// in a single call. This is the fast path — one C call replaces 20+ Go calls.
    struct Line { int x1, y1, x2, y2; uint8_t r, g, b; int thickness; };
    struct TextOverlay { int x, y; char text[128]; uint8_t r, g, b; int scale; };
    struct FilledRect { int x1, y1, x2, y2; uint8_t r, g, b, alpha; };
    struct PolygonOverlay { Point pts[8]; int n_pts; uint8_t r, g, b, alpha; };

    static uint8_t* annotate_full(
        const uint8_t* rgb, int w, int h,
        const AnnotateBox* boxes, int n_boxes,
        const Line* lines, int n_lines,
        const PolygonOverlay* polygons, int n_polygons,
        const TextOverlay* texts, int n_texts,
        const FilledRect* rects, int n_rects,
        int out_w, int out_h, int quality, int* out_size);
};

} // namespace infergo
