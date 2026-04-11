// cpp/video/frame_annotator_test.cpp
// GTest tests for FrameAnnotator.

#include "frame_annotator.hpp"
#include <gtest/gtest.h>
#include <cstring>
#include <vector>

using namespace infergo;

// Helper: create a blank (black) RGB buffer.
static std::vector<uint8_t> make_rgb(int w, int h) {
    return std::vector<uint8_t>(static_cast<size_t>(w) * h * 3, 0);
}

// Helper: read pixel at (x, y).
static void get_pixel(const uint8_t* rgb, int w, int /*h*/,
                      int x, int y,
                      uint8_t& r, uint8_t& g, uint8_t& b) {
    int off = (y * w + x) * 3;
    r = rgb[off + 0];
    g = rgb[off + 1];
    b = rgb[off + 2];
}

TEST(FrameAnnotator, DrawRect) {
    int w = 100, h = 100;
    auto buf = make_rgb(w, h);

    // Draw a red rect from (10,10) to (90,90) with 2px thickness
    FrameAnnotator::draw_rect(buf.data(), w, h, 10, 10, 90, 90, 255, 0, 0, 2);

    // Check top-left corner is red
    uint8_t r, g, b;
    get_pixel(buf.data(), w, h, 10, 10, r, g, b);
    EXPECT_EQ(r, 255);
    EXPECT_EQ(g, 0);
    EXPECT_EQ(b, 0);

    // Check bottom-right corner is red
    get_pixel(buf.data(), w, h, 90, 90, r, g, b);
    EXPECT_EQ(r, 255);
    EXPECT_EQ(g, 0);
    EXPECT_EQ(b, 0);

    // Check a point well inside the rect is still black
    get_pixel(buf.data(), w, h, 50, 50, r, g, b);
    EXPECT_EQ(r, 0);
    EXPECT_EQ(g, 0);
    EXPECT_EQ(b, 0);
}

TEST(FrameAnnotator, DrawText) {
    int w = 200, h = 50;
    auto buf = make_rgb(w, h);

    FrameAnnotator::draw_text(buf.data(), w, h, 10, 10, "Hi", 255, 255, 255, 1);

    // Verify some non-zero pixels exist in the text region
    bool found = false;
    for (int y = 10; y < 20; ++y) {
        for (int x = 10; x < 30; ++x) {
            uint8_t r, g, b;
            get_pixel(buf.data(), w, h, x, y, r, g, b);
            if (r > 0 || g > 0 || b > 0) {
                found = true;
                break;
            }
        }
        if (found) break;
    }
    EXPECT_TRUE(found) << "Expected non-zero pixels in text region";
}

TEST(FrameAnnotator, DrawFilledRect) {
    int w = 100, h = 100;
    auto buf = make_rgb(w, h);

    // Fill a rect with opaque green
    FrameAnnotator::draw_filled_rect(buf.data(), w, h, 20, 20, 40, 40, 0, 255, 0, 255);

    uint8_t r, g, b;
    get_pixel(buf.data(), w, h, 30, 30, r, g, b);
    EXPECT_EQ(r, 0);
    EXPECT_EQ(g, 255);
    EXPECT_EQ(b, 0);
}

TEST(FrameAnnotator, DrawLine) {
    int w = 100, h = 100;
    auto buf = make_rgb(w, h);

    FrameAnnotator::draw_line(buf.data(), w, h, 0, 0, 99, 99, 255, 128, 0, 1);

    // The diagonal should have some non-zero pixels
    uint8_t r, g, b;
    get_pixel(buf.data(), w, h, 50, 50, r, g, b);
    EXPECT_GT(r, 0);
}

#ifdef INFER_TURBOJPEG_AVAILABLE
TEST(FrameAnnotator, AnnotateJPEG) {
    int w = 320, h = 240;
    auto buf = make_rgb(w, h);
    // Fill with some non-zero data so JPEG encoding is meaningful
    for (size_t i = 0; i < buf.size(); i += 3) {
        buf[i + 0] = 100;
        buf[i + 1] = 150;
        buf[i + 2] = 200;
    }

    AnnotateBox boxes[1];
    boxes[0].x1 = 50;
    boxes[0].y1 = 50;
    boxes[0].x2 = 200;
    boxes[0].y2 = 180;
    boxes[0].class_id = 0;
    boxes[0].confidence = 0.95f;
    boxes[0].track_id = 1;
    std::strncpy(boxes[0].label, "person #1", sizeof(boxes[0].label));

    int jpeg_size = 0;
    uint8_t* jpeg = FrameAnnotator::annotate_jpeg(buf.data(), w, h, boxes, 1, 75, &jpeg_size);

    ASSERT_NE(jpeg, nullptr);
    EXPECT_GT(jpeg_size, 2);
    // Check JPEG magic bytes
    EXPECT_EQ(jpeg[0], 0xFF);
    EXPECT_EQ(jpeg[1], 0xD8);

    FrameAnnotator::jpeg_free(jpeg);
}

TEST(FrameAnnotator, ResizeJPEG) {
    int sw = 640, sh = 480;
    auto buf = make_rgb(sw, sh);
    for (size_t i = 0; i < buf.size(); i += 3) {
        buf[i] = 80;
        buf[i + 1] = 120;
        buf[i + 2] = 160;
    }

    int jpeg_size = 0;
    uint8_t* jpeg = FrameAnnotator::resize_jpeg(buf.data(), sw, sh, 320, 240, 75, &jpeg_size);

    ASSERT_NE(jpeg, nullptr);
    EXPECT_GT(jpeg_size, 2);
    EXPECT_EQ(jpeg[0], 0xFF);
    EXPECT_EQ(jpeg[1], 0xD8);

    FrameAnnotator::jpeg_free(jpeg);
}

TEST(FrameAnnotator, CombineJPEG) {
    int w = 320, h = 240;
    auto buf1 = make_rgb(w, h);
    auto buf2 = make_rgb(w, h);
    // Fill with distinct colors
    for (size_t i = 0; i < buf1.size(); i += 3) {
        buf1[i] = 200; buf1[i + 1] = 50; buf1[i + 2] = 50;
    }
    for (size_t i = 0; i < buf2.size(); i += 3) {
        buf2[i] = 50; buf2[i + 1] = 50; buf2[i + 2] = 200;
    }

    int jpeg_size = 0;
    uint8_t* jpeg = FrameAnnotator::combine_jpeg(
        buf1.data(), w, h, buf2.data(), w, h,
        "Frame 42 | FPS: 30.0", 640, 520, 75, &jpeg_size);

    ASSERT_NE(jpeg, nullptr);
    EXPECT_GT(jpeg_size, 2);
    EXPECT_EQ(jpeg[0], 0xFF);
    EXPECT_EQ(jpeg[1], 0xD8);

    FrameAnnotator::jpeg_free(jpeg);
}
#endif  // INFER_TURBOJPEG_AVAILABLE
