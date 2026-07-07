/* vipnet.h - minimal VIPLite v2.0 wrapper: load an NBG, feed inputs, run, read
 * outputs as fp32. One struct per network, no globals except the driver init.
 *
 * The dequantization follows the NBG metadata (dynamic fixed point, fl):
 *   fp32 = int16 / 2^fl        int16 = round(fp32 * 2^fl), clamped
 */
#ifndef VIPNET_H
#define VIPNET_H

#include <stdint.h>
#include "vip_lite.h"

#define VIPNET_MAX_IO 4

typedef struct {
    vip_buffer_create_params_t p;   /* dims, format, quantization (queried) */
    vip_buffer buf;
    uint32_t elements;              /* product of dims */
    char name[64];
} vipnet_io_t;

typedef struct {
    vip_network net;
    vipnet_io_t in[VIPNET_MAX_IO];
    vipnet_io_t out[VIPNET_MAX_IO];
    uint32_t n_in, n_out;
} vipnet_t;

/* Driver-wide init/exit (vip_init / vip_destroy). Call once per process. */
int  vipnet_global_init(void);
void vipnet_global_exit(void);

/* Load network_binary.nb, create+attach I/O buffers, prepare for run.
 * Returns 0 on success. */
int  vipnet_open(vipnet_t *n, const char *nbg_path);
void vipnet_close(vipnet_t *n);

/* Write input i from fp32 data (handles int16-dfp and fp16 inputs), flush. */
int vipnet_write_input_fp32(vipnet_t *n, int i, const float *src, uint32_t count);

/* Zero-copy path: map input i for direct writing (returns NULL on failure),
 * then commit (unmap + cache flush) before running. Lets the preprocessing
 * quantize straight into the NPU buffer, no intermediate float pass. */
void *vipnet_input_map(vipnet_t *n, int i);
void  vipnet_input_commit(vipnet_t *n, int i);

/* Run one inference (blocking). */
int vipnet_run(vipnet_t *n);

/* Read output i into fp32 (handles int16-dfp, fp16, fp32), after cache
 * invalidate. Returns number of elements read, or -1. */
int vipnet_read_output_fp32(vipnet_t *n, int i, float *dst, uint32_t max_count);

/* fp16 helpers (IEEE 754 half) */
uint16_t fl_fp32_to_fp16(float f);
float    fl_fp16_to_fp32(uint16_t h);

#endif /* VIPNET_H */
