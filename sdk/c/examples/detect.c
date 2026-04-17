// infergo C SDK — Detection example
#include "infer_api.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <model.onnx> <image.jpg>\n", argv[0]); return 1; }
    InferSession s = infer_session_create("cpu", 0);
    if (!s) { fprintf(stderr, "Session failed: %s\n", infer_last_error_string()); return 1; }
    if (infer_session_load(s, argv[1]) != 0) { fprintf(stderr, "Load failed: %s\n", infer_last_error_string()); infer_session_destroy(s); return 1; }

    FILE* f = fopen(argv[2], "rb");
    if (!f) { perror("fopen"); infer_session_destroy(s); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void* data = malloc(sz); fread(data, 1, sz, f); fclose(f);

    InferTensor img = infer_preprocess_decode_image(data, (int)sz);
    free(data);
    if (!img) { fprintf(stderr, "Decode failed: %s\n", infer_last_error_string()); infer_session_destroy(s); return 1; }

    InferTensor resized = infer_preprocess_letterbox(img, 640, 640);
    infer_tensor_free(img);
    if (!resized) { fprintf(stderr, "Letterbox failed\n"); infer_session_destroy(s); return 1; }

    float mean[] = {0,0,0}, std[] = {1,1,1};
    InferTensor norm = infer_preprocess_normalize(resized, 255.0f, mean, std);
    infer_tensor_free(resized);
    if (!norm) { fprintf(stderr, "Normalize failed\n"); infer_session_destroy(s); return 1; }

    InferTensor batch = infer_preprocess_stack_batch(&norm, 1);
    infer_tensor_free(norm);
    if (!batch) { fprintf(stderr, "Stack failed\n"); infer_session_destroy(s); return 1; }

    InferTensor out = NULL;
    if (infer_session_run(s, &batch, 1, &out, 1) != 0) {
        fprintf(stderr, "Inference failed: %s\n", infer_last_error_string());
        infer_tensor_free(batch); infer_session_destroy(s); return 1;
    }

    InferBox boxes[100];
    int n = infer_postprocess_nms(out, 0.25f, 0.45f, boxes, 100);
    for (int i = 0; i < n; i++)
        printf("class=%d conf=%.2f (%.0f,%.0f)-(%.0f,%.0f)\n",
               boxes[i].class_idx, boxes[i].confidence, boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2);
    infer_tensor_free(out); infer_tensor_free(batch);
    infer_session_destroy(s);
    return 0;
}
