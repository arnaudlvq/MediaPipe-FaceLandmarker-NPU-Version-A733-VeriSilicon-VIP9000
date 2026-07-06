/* facelandmarker.c - the chained pipeline.
 *
 * Frame (RGB888)
 *   -> letterbox 128x128, [-1,1]        (CPU)
 *   -> face_detector NBG                 (NPU)  896 anchors: boxes + 6 keypoints
 *   -> decode + best detection           (CPU)  sigmoid, threshold, top-1
 *   -> rotated square crop 256x256,[0,1] (CPU)  eye-line rotation, 1.5x scale
 *   -> face_landmarks_detector NBG       (NPU)  478 x (x,y,z) + presence
 *   -> map landmarks back to frame px    (CPU)
 *   -> subset 146 points (pixel coords)  (CPU)
 *   -> face_blendshapes NBG              (NPU)  52 scores
 *
 * The CPU stages replicate the MediaPipe calculators the .task graph uses:
 *   SsdAnchorsCalculator        -> gen_anchors()          (896 fixed anchors)
 *   TensorsToDetectionsCalculator -> decode_best()        (x/y/w/h scale 128)
 *   DetectionsToRectsCalculator -> rotation from keypoints 0-1, target 0 rad
 *   RectTransformationCalculator -> scale 1.5, square-long
 *   LandmarksToTensorCalculator -> subset x img_size      (pixel coordinates;
 *     the blendshapes MLP-Mixer standardizes its input internally, which is
 *     why plain pixel coordinates are the expected input)
 *
 * Simplifications vs the full graph, documented in runner/README.md:
 *   - top-1 detection instead of weighted NMS (single-viewer use case)
 *   - detector runs every frame (0.7 ms on the NPU; no tracking state)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "vipnet.h"
#include "facelandmarker.h"

/* ---------------- constants from the model + graph configs ---------------- */
#define DET_IN     128
#define LMK_IN     256
#define N_ANCHORS  896
#define DET_SCORE_MIN   0.5f     /* min_detection_confidence */
#define PRESENCE_MIN    0.5f     /* min_face_presence_confidence */
#define ROI_SCALE       1.5f     /* RectTransformation scale_x/scale_y */

const char *FL_BLENDSHAPE_NAMES[FL_NUM_BLENDSHAPES] = {
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
};

/* The 146-landmark subset the blendshapes model takes as input.
 * Verbatim kLandmarksSubsetIdxs from MediaPipe face_blendshapes_graph.cc. */
static const int SUBSET146[146] = {
      0,   1,   4,   5,   6,   7,   8,  10,  13,  14,  17,  21,  33,  37,  39,
     40,  46,  52,  53,  54,  55,  58,  61,  63,  65,  66,  67,  70,  78,  80,
     81,  82,  84,  87,  88,  91,  93,  95, 103, 105, 107, 109, 127, 132, 133,
    136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
    161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
    249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
    296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
    336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
    466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
};

typedef struct { float cx, cy; } anchor_t;

struct fl_ctx {
    vipnet_t det, lmk, bs;
    anchor_t anchors[N_ANCHORS];
    /* scratch */
    float det_in[DET_IN * DET_IN * 3];
    float lmk_in[LMK_IN * LMK_IN * 3];
    float regressors[N_ANCHORS * 16];
    float scores[N_ANCHORS];
    float lmk_raw[FL_NUM_LANDMARKS * 3];
    float bs_in[146 * 2];
};

/* SsdAnchorsCalculator for the short-range BlazeFace config:
 * input 128, strides {8,16,16,16}, fixed_anchor_size, offset 0.5.
 * Layer 0: 16x16 cells x 2 anchors = 512; layers 1-3 share stride 16 and
 * merge into 8x8 cells x 6 anchors = 384. Total 896. */
static void gen_anchors(anchor_t *a)
{
    int n = 0, y, x, k;
    for (y = 0; y < 16; y++)
        for (x = 0; x < 16; x++)
            for (k = 0; k < 2; k++, n++) {
                a[n].cx = (x + 0.5f) / 16.f;
                a[n].cy = (y + 0.5f) / 16.f;
            }
    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++)
            for (k = 0; k < 6; k++, n++) {
                a[n].cx = (x + 0.5f) / 8.f;
                a[n].cy = (y + 0.5f) / 8.f;
            }
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static float sigmoidf(float v)
{
    if (v < -100.f) v = -100.f;
    if (v >  100.f) v =  100.f;
    return 1.f / (1.f + expf(-v));
}

/* Bilinear sample of an RGB888 image, zero outside. Returns one channel. */
static inline float sample_rgb(const unsigned char *rgb, int w, int h,
                               float fx, float fy, int c)
{
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float ax = fx - x0, ay = fy - y0;
    float v = 0.f;
    int dx, dy;
    for (dy = 0; dy <= 1; dy++)
        for (dx = 0; dx <= 1; dx++) {
            int xx = x0 + dx, yy = y0 + dy;
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
            float wgt = (dx ? ax : 1.f - ax) * (dy ? ay : 1.f - ay);
            v += wgt * rgb[(yy * w + xx) * 3 + c];
        }
    return v;
}

/* Letterbox the frame into dst (side x side x 3 fp32), values [lo, hi].
 * Writes the mapping (scale, pad) needed to project detections back. */
static void letterbox(const unsigned char *rgb, int w, int h,
                      float *dst, int side, float lo, float hi,
                      float *scale_out, float *padx_out, float *pady_out)
{
    float scale = (float)side / (float)(w > h ? w : h);
    float outw = w * scale, outh = h * scale;
    float padx = (side - outw) * 0.5f, pady = (side - outh) * 0.5f;
    float k = (hi - lo) / 255.f;
    int x, y, c;
    for (y = 0; y < side; y++) {
        for (x = 0; x < side; x++) {
            float sx = (x + 0.5f - padx) / scale - 0.5f;
            float sy = (y + 0.5f - pady) / scale - 0.5f;
            float *px = dst + (y * side + x) * 3;
            if (sx < -1.f || sy < -1.f || sx > w || sy > h) {
                px[0] = px[1] = px[2] = lo;
            } else {
                for (c = 0; c < 3; c++)
                    px[c] = lo + k * sample_rgb(rgb, w, h, sx, sy, c);
            }
        }
    }
    *scale_out = scale; *padx_out = padx; *pady_out = pady;
}

/* Decode the 896 anchors, return index of the best-scoring detection
 * (or -1), with its box and 6 keypoints in letterbox-normalized coords. */
static int decode_best(const float *reg, const float *logits,
                       const anchor_t *anchors,
                       float *score, float box[4], float kp[12])
{
    int best = -1, i, j;
    float best_s = DET_SCORE_MIN;
    for (i = 0; i < N_ANCHORS; i++) {
        float s = sigmoidf(logits[i]);
        if (s <= best_s) continue;
        best_s = s; best = i;
    }
    if (best < 0) return -1;
    {
        const float *r = reg + best * 16;
        float acx = anchors[best].cx, acy = anchors[best].cy;
        float cx = r[0] / DET_IN + acx;
        float cy = r[1] / DET_IN + acy;
        float w  = r[2] / DET_IN;
        float h  = r[3] / DET_IN;
        box[0] = cx; box[1] = cy; box[2] = w; box[3] = h;
        for (j = 0; j < 6; j++) {
            kp[j * 2 + 0] = r[4 + j * 2 + 0] / DET_IN + acx;
            kp[j * 2 + 1] = r[4 + j * 2 + 1] / DET_IN + acy;
        }
    }
    *score = best_s;
    return best;
}

fl_ctx_t *fl_create(const char *models_dir)
{
    static const char *sub[3] = {
        "face_detector_nbg_int16", "face_landmarks_detector_nbg_int16",
        "face_blendshapes_nbg_int16",
    };
    fl_ctx_t *ctx;
    vipnet_t *nets[3];
    char path[1024];
    int i;

    if (vipnet_global_init() != 0) return NULL;
    ctx = (fl_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    nets[0] = &ctx->det; nets[1] = &ctx->lmk; nets[2] = &ctx->bs;
    for (i = 0; i < 3; i++) {
        snprintf(path, sizeof(path), "%s/%s/network_binary.nb", models_dir, sub[i]);
        if (vipnet_open(nets[i], path) != 0) {
            fprintf(stderr, "fl_create: cannot open %s\n", path);
            fl_destroy(ctx);
            return NULL;
        }
    }
    gen_anchors(ctx->anchors);
    return ctx;
}

void fl_destroy(fl_ctx_t *ctx)
{
    if (!ctx) return;
    vipnet_close(&ctx->det);
    vipnet_close(&ctx->lmk);
    vipnet_close(&ctx->bs);
    free(ctx);
    vipnet_global_exit();
}

int fl_process_rgb(fl_ctx_t *ctx, const unsigned char *rgb, int width, int height,
                   fl_result_t *out)
{
    double t0 = now_ms(), t1, t3, t4;
    float scale, padx, pady;
    float score, box[4], kp[12];
    int i;

    memset(out, 0, sizeof(*out));

    /* -------- stage 1: detector -------- */
    letterbox(rgb, width, height, ctx->det_in, DET_IN, -1.f, 1.f,
              &scale, &padx, &pady);
    if (vipnet_write_input_fp32(&ctx->det, 0, ctx->det_in, DET_IN * DET_IN * 3) != 0)
        return -1;
    if (vipnet_run(&ctx->det) != 0) return -1;
    /* output order per nbg_meta.json: regressors [896x16], classificators [896] */
    if (vipnet_read_output_fp32(&ctx->det, 0, ctx->regressors, N_ANCHORS * 16) < 0)
        return -1;
    if (vipnet_read_output_fp32(&ctx->det, 1, ctx->scores, N_ANCHORS) < 0)
        return -1;
    t1 = now_ms();
    out->ms_detect = t1 - t0;

    if (decode_best(ctx->regressors, ctx->scores, ctx->anchors,
                    &score, box, kp) < 0) {
        out->ms_total = now_ms() - t0;
        return 0;                       /* no face */
    }
    out->detect_score = score;

    /* -------- stage 2: ROI + landmarks -------- */
    {
        /* letterbox-normalized -> frame pixels */
        float fx = (box[0] * DET_IN - padx) / scale;
        float fy = (box[1] * DET_IN - pady) / scale;
        float fw = box[2] * DET_IN / scale;
        float fh = box[3] * DET_IN / scale;
        /* rotation from keypoint 0 (right eye) to 1 (left eye), target 0 rad */
        float ex0 = (kp[0] * DET_IN - padx) / scale, ey0 = (kp[1] * DET_IN - pady) / scale;
        float ex1 = (kp[2] * DET_IN - padx) / scale, ey1 = (kp[3] * DET_IN - pady) / scale;
        float rot = atan2f(ey1 - ey0, ex1 - ex0);
        float side = ROI_SCALE * (fw > fh ? fw : fh);   /* square-long * 1.5 */
        float cr = cosf(rot), sr = sinf(rot);
        int x, y, c;

        out->roi_cx = fx; out->roi_cy = fy;
        out->roi_size = side; out->roi_rotation = rot;

        /* rotated crop -> 256x256, [0,1] */
        for (y = 0; y < LMK_IN; y++) {
            for (x = 0; x < LMK_IN; x++) {
                float u = ((x + 0.5f) / LMK_IN - 0.5f) * side;
                float v = ((y + 0.5f) / LMK_IN - 0.5f) * side;
                float sx = fx + u * cr - v * sr - 0.5f;
                float sy = fy + u * sr + v * cr - 0.5f;
                float *px = ctx->lmk_in + (y * LMK_IN + x) * 3;
                for (c = 0; c < 3; c++)
                    px[c] = sample_rgb(rgb, width, height, sx, sy, c) / 255.f;
            }
        }

        if (vipnet_write_input_fp32(&ctx->lmk, 0, ctx->lmk_in,
                                    LMK_IN * LMK_IN * 3) != 0) return -1;
        if (vipnet_run(&ctx->lmk) != 0) return -1;
        /* outputs: Identity [1434] landmarks, Identity_1 [1] presence logit */
        if (vipnet_read_output_fp32(&ctx->lmk, 0, ctx->lmk_raw,
                                    FL_NUM_LANDMARKS * 3) < 0) return -1;
        {
            float presence = 0.f;
            vipnet_read_output_fp32(&ctx->lmk, 1, &presence, 1);
            out->presence_score = sigmoidf(presence);
        }
        t3 = now_ms();
        out->ms_landmarks = t3 - t1;

        /* crop coords (0..256) -> frame pixels */
        for (i = 0; i < FL_NUM_LANDMARKS; i++) {
            float lx = ctx->lmk_raw[i * 3 + 0] / LMK_IN - 0.5f;
            float ly = ctx->lmk_raw[i * 3 + 1] / LMK_IN - 0.5f;
            float lz = ctx->lmk_raw[i * 3 + 2] / LMK_IN;
            out->landmarks[i].x = fx + (lx * cr - ly * sr) * side;
            out->landmarks[i].y = fy + (lx * sr + ly * cr) * side;
            out->landmarks[i].z = lz * side;
        }
    }

    if (out->presence_score < PRESENCE_MIN) {
        out->ms_total = now_ms() - t0;
        return 0;                       /* box found but no reliable face */
    }
    out->face_present = 1;

    /* -------- stage 3: blendshapes -------- */
    for (i = 0; i < 146; i++) {
        ctx->bs_in[i * 2 + 0] = out->landmarks[SUBSET146[i]].x;
        ctx->bs_in[i * 2 + 1] = out->landmarks[SUBSET146[i]].y;
    }
    if (vipnet_write_input_fp32(&ctx->bs, 0, ctx->bs_in, 146 * 2) != 0) return -1;
    if (vipnet_run(&ctx->bs) != 0) return -1;
    if (vipnet_read_output_fp32(&ctx->bs, 0, out->blendshapes,
                                FL_NUM_BLENDSHAPES) < 0) return -1;
    t4 = now_ms();
    out->ms_blendshapes = t4 - t3;
    out->ms_total = t4 - t0;
    return 0;
}
