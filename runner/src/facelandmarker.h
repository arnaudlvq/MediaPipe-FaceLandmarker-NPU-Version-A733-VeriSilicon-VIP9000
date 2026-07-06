/* facelandmarker.h - public C API of the chained NPU runner.
 *
 * One call per frame: RGB888 in, 478 landmarks + 52 blendshapes out.
 * The three MediaPipe FaceLandmarker models run on the VIP9000 NPU (int16
 * NBG); the glue between them (letterbox, anchor decode, crop, landmark
 * subset) runs on the CPU in this library.
 *
 * Also builds as a shared library (libfacelandmarker_npu.so) so any language
 * with a C FFI (Python ctypes, etc.) can drive the NPU pipeline directly.
 */
#ifndef FACELANDMARKER_H
#define FACELANDMARKER_H

#ifdef __cplusplus
extern "C" {
#endif

#define FL_NUM_LANDMARKS   478
#define FL_NUM_BLENDSHAPES 52

typedef struct {
    float x, y, z;              /* x,y in input-frame pixels; z same scale as x */
} fl_point3_t;

typedef struct {
    int   face_present;         /* 1 if a face was detected AND tracked */
    float detect_score;         /* BlazeFace confidence, 0..1 */
    float presence_score;       /* landmark-model face presence, 0..1 */
    fl_point3_t landmarks[FL_NUM_LANDMARKS];
    float blendshapes[FL_NUM_BLENDSHAPES];   /* 0..1, order = FL_BLENDSHAPE_NAMES */
    /* face ROI used for the landmark crop (debug / reuse) */
    float roi_cx, roi_cy, roi_size, roi_rotation;
    /* per-stage wall time, milliseconds */
    double ms_detect, ms_landmarks, ms_blendshapes, ms_total;
} fl_result_t;

typedef struct fl_ctx fl_ctx_t;

/* models_dir must contain:
 *   face_detector_nbg_int16/network_binary.nb
 *   face_landmarks_detector_nbg_int16/network_binary.nb
 *   face_blendshapes_nbg_int16/network_binary.nb
 * Returns NULL on failure (details on stderr). */
fl_ctx_t *fl_create(const char *models_dir);

/* Process one RGB888 frame (w*h*3 bytes, row-major, no padding).
 * Returns 0 on success (even when no face: face_present tells). */
int fl_process_rgb(fl_ctx_t *ctx, const unsigned char *rgb, int width, int height,
                   fl_result_t *out);

void fl_destroy(fl_ctx_t *ctx);

extern const char *FL_BLENDSHAPE_NAMES[FL_NUM_BLENDSHAPES];

#ifdef __cplusplus
}
#endif
#endif /* FACELANDMARKER_H */
