/* IRD detector: C API (the integration interface for the host system).
 *
 * One detector = one TensorRT engine + its runtime config (<model>.onnx.runtime.txt).
 * Each call processes one frame on its own (RCI: frames are independent).
 *
 *   ird_detector* det = NULL;
 *   ird_options opt = ird_default_options();
 *   if (ird_create("a_ir.engine", "a_ir.onnx.runtime.txt", &opt, &det) != IRD_OK) puts(ird_last_error());
 *   ird_image ir = { pixels, 640, 512, 1, 14 };            // raw 14-bit frame
 *   ird_detection out[300]; int n = 0; ird_timing t;
 *   ird_detect(det, &ir, NULL, out, 300, &n, &t);
 *   ird_destroy(det);
 */
#ifndef IRD_API_H
#define IRD_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IRD_OK 0
#define IRD_ERR_ARG 1
#define IRD_ERR_FILE 2
#define IRD_ERR_ENGINE 3
#define IRD_ERR_CUDA 4
#define IRD_MAX_CLASSES 16

typedef struct ird_detector ird_detector;

typedef struct {
    const uint16_t* data;  /* interleaved HWC, row-major */
    int width, height;
    int channels;          /* 1 (IR) or 3 (visible) */
    int bit_depth;         /* 8..16 */
} ird_image;

typedef struct {
    int32_t class_id;      /* index into the class list of the runtime config */
    float x, y, w, h;      /* top-left corner and size, original image pixels */
    float score;           /* 0..1 */
} ird_detection;

typedef struct {
    float score_threshold;                   /* detections below are dropped (default 0.5) */
    float class_thresholds[IRD_MAX_CLASSES]; /* per class; < 0 means use score_threshold */
    int max_detections;                      /* default 300 */
    int use_cuda_graph;                      /* 1: capture copy+inference in a CUDA graph (default 1) */
    int top_k;                               /* candidates before thresholding (default 300) */
} ird_options;

typedef struct {
    double preprocess_ms;  /* CPU: tone map, channels, padding */
    double infer_ms;       /* host->device copy, TensorRT, device->host copy */
    double decode_ms;      /* sigmoid, top-k, boxes, thresholds */
    double total_ms;       /* frame available -> detections ready */
} ird_timing;

ird_options ird_default_options(void);

int ird_create(const char* engine_path, const char* runtime_cfg_path, const ird_options* options,
               ird_detector** out);

/* modality 'ir' needs ir; 'vis' needs vis; 'fusion' needs both (same size). Unused may be NULL. */
int ird_detect(ird_detector* det, const ird_image* ir, const ird_image* vis, ird_detection* out,
               int capacity, int* n_out, ird_timing* timing);

void ird_destroy(ird_detector* det);

const char* ird_last_error(void);
int ird_num_classes(const ird_detector* det);
const char* ird_class_name(const ird_detector* det, int class_id);
const char* ird_modality(const ird_detector* det);

#ifdef __cplusplus
}
#endif
#endif /* IRD_API_H */
