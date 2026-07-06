/* vipnet.c - minimal VIPLite v2.0 wrapper. See vipnet.h.
 *
 * The call sequence mirrors the vendor examples (ai-sdk vpm_run and
 * libawnn_viplite): vip_init -> vip_create_network(FROM_FILE) -> query I/O
 * properties -> vip_create_buffer -> vip_prepare_network -> vip_set_input/
 * output -> per frame: map+memcpy+flush(FLUSH) -> vip_run_network ->
 * flush(INVALIDATE)+map+dequantize.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vipnet.h"

#define CHECK(st, msg) \
    do { if ((st) != VIP_SUCCESS) { fprintf(stderr, "vipnet: %s failed, status=%d\n", msg, (int)(st)); return -1; } } while (0)

int vipnet_global_init(void)
{
    vip_status_e st = vip_init();
    CHECK(st, "vip_init");
    return 0;
}

void vipnet_global_exit(void)
{
    vip_destroy();
}

static int query_io(vipnet_t *n, int is_input, uint32_t idx, vipnet_io_t *io)
{
    vip_buffer_create_params_t *p = &io->p;
    vip_status_e st;
    uint32_t d;

    memset(p, 0, sizeof(*p));
#define Q(prop, dst) \
    st = is_input ? vip_query_input(n->net, idx, prop, dst) \
                  : vip_query_output(n->net, idx, prop, dst); \
    CHECK(st, #prop)
    Q(VIP_BUFFER_PROP_DATA_FORMAT,      &p->data_format);
    Q(VIP_BUFFER_PROP_NUM_OF_DIMENSION, &p->num_of_dims);
    Q(VIP_BUFFER_PROP_SIZES_OF_DIMENSION, p->sizes);
    Q(VIP_BUFFER_PROP_QUANT_FORMAT,     &p->quant_format);
    Q(VIP_BUFFER_PROP_NAME,             io->name);
    if (p->quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        Q(VIP_BUFFER_PROP_FIXED_POINT_POS, &p->quant_data.dfp.fixed_point_pos);
    } else if (p->quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) {
        Q(VIP_BUFFER_PROP_TF_SCALE,      &p->quant_data.affine.scale);
        Q(VIP_BUFFER_PROP_TF_ZERO_POINT, &p->quant_data.affine.zeroPoint);
    }
#undef Q
    p->memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;

    io->elements = 1;
    for (d = 0; d < p->num_of_dims; d++)
        io->elements *= p->sizes[d];

    st = vip_create_buffer(p, sizeof(*p), &io->buf);
    CHECK(st, "vip_create_buffer");
    return 0;
}

int vipnet_open(vipnet_t *n, const char *nbg_path)
{
    vip_status_e st;
    uint32_t i;

    memset(n, 0, sizeof(*n));
    st = vip_create_network(nbg_path, 0, VIP_CREATE_NETWORK_FROM_FILE, &n->net);
    CHECK(st, "vip_create_network");

    st = vip_query_network(n->net, VIP_NETWORK_PROP_INPUT_COUNT, &n->n_in);
    CHECK(st, "query input count");
    st = vip_query_network(n->net, VIP_NETWORK_PROP_OUTPUT_COUNT, &n->n_out);
    CHECK(st, "query output count");
    if (n->n_in > VIPNET_MAX_IO || n->n_out > VIPNET_MAX_IO) {
        fprintf(stderr, "vipnet: too many I/O (%u in, %u out)\n", n->n_in, n->n_out);
        return -1;
    }

    for (i = 0; i < n->n_in; i++)
        if (query_io(n, 1, i, &n->in[i]) != 0) return -1;
    for (i = 0; i < n->n_out; i++)
        if (query_io(n, 0, i, &n->out[i]) != 0) return -1;

    st = vip_prepare_network(n->net);
    CHECK(st, "vip_prepare_network");

    for (i = 0; i < n->n_in; i++) {
        st = vip_set_input(n->net, i, n->in[i].buf);
        CHECK(st, "vip_set_input");
    }
    for (i = 0; i < n->n_out; i++) {
        st = vip_set_output(n->net, i, n->out[i].buf);
        CHECK(st, "vip_set_output");
    }
    return 0;
}

void vipnet_close(vipnet_t *n)
{
    uint32_t i;
    if (n->net) {
        vip_finish_network(n->net);
        vip_destroy_network(n->net);
    }
    for (i = 0; i < n->n_in; i++)
        if (n->in[i].buf) vip_destroy_buffer(n->in[i].buf);
    for (i = 0; i < n->n_out; i++)
        if (n->out[i].buf) vip_destroy_buffer(n->out[i].buf);
    memset(n, 0, sizeof(*n));
}

/* ---- fp16 (IEEE 754 half, round to nearest even not needed here: truncate
 * with rounding bit is within 1 ulp, fine for coordinates) ---- */
uint16_t fl_fp32_to_fp16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man  = x & 0x007FFFFFu;
    if (exp <= 0) {                       /* underflow -> signed zero/denorm */
        if (exp < -10) return (uint16_t)sign;
        man |= 0x00800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        return (uint16_t)(sign | ((man + (1u << (shift - 1))) >> shift));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);   /* overflow -> inf */
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((man + 0x1000u) >> 13));
}

float fl_fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x03FFu;
    uint32_t x;
    float f;
    if (exp == 0) {
        if (man == 0) { x = sign; }
        else {                                  /* subnormal */
            exp = 1;
            while (!(man & 0x0400u)) { man <<= 1; exp--; }
            man &= 0x03FFu;
            x = sign | ((exp + 127 - 15) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7F800000u | (man << 13);
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    memcpy(&f, &x, 4);
    return f;
}

int vipnet_write_input_fp32(vipnet_t *n, int i, const float *src, uint32_t count)
{
    vipnet_io_t *io = &n->in[i];
    void *dst = vip_map_buffer(io->buf);
    uint32_t k;
    if (!dst) { fprintf(stderr, "vipnet: map input %d failed\n", i); return -1; }
    if (count > io->elements) count = io->elements;

    if (io->p.data_format == VIP_BUFFER_FORMAT_INT16 &&
        io->p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        int16_t *d = (int16_t *)dst;
        float s = (float)(1 << io->p.quant_data.dfp.fixed_point_pos);
        for (k = 0; k < count; k++) {
            float v = src[k] * s;
            if (v >  32767.f) v =  32767.f;
            if (v < -32768.f) v = -32768.f;
            d[k] = (int16_t)lrintf(v);
        }
    } else if (io->p.data_format == VIP_BUFFER_FORMAT_FP16) {
        uint16_t *d = (uint16_t *)dst;
        for (k = 0; k < count; k++) d[k] = fl_fp32_to_fp16(src[k]);
    } else if (io->p.data_format == VIP_BUFFER_FORMAT_FP32) {
        memcpy(dst, src, count * 4);
    } else {
        fprintf(stderr, "vipnet: unsupported input format %d\n", (int)io->p.data_format);
        vip_unmap_buffer(io->buf);
        return -1;
    }
    vip_unmap_buffer(io->buf);
    vip_flush_buffer(io->buf, VIP_BUFFER_OPER_TYPE_FLUSH);
    return 0;
}

int vipnet_run(vipnet_t *n)
{
    vip_status_e st = vip_run_network(n->net);
    CHECK(st, "vip_run_network");
    return 0;
}

int vipnet_read_output_fp32(vipnet_t *n, int i, float *dst, uint32_t max_count)
{
    vipnet_io_t *io = &n->out[i];
    uint32_t count = io->elements < max_count ? io->elements : max_count;
    uint32_t k;
    void *srcp;

    vip_flush_buffer(io->buf, VIP_BUFFER_OPER_TYPE_INVALIDATE);
    srcp = vip_map_buffer(io->buf);
    if (!srcp) { fprintf(stderr, "vipnet: map output %d failed\n", i); return -1; }

    if (io->p.data_format == VIP_BUFFER_FORMAT_INT16 &&
        io->p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        const int16_t *s = (const int16_t *)srcp;
        float inv = 1.f / (float)(1 << io->p.quant_data.dfp.fixed_point_pos);
        for (k = 0; k < count; k++) dst[k] = s[k] * inv;
    } else if (io->p.data_format == VIP_BUFFER_FORMAT_FP16) {
        const uint16_t *s = (const uint16_t *)srcp;
        for (k = 0; k < count; k++) dst[k] = fl_fp16_to_fp32(s[k]);
    } else if (io->p.data_format == VIP_BUFFER_FORMAT_FP32) {
        memcpy(dst, srcp, count * 4);
    } else {
        fprintf(stderr, "vipnet: unsupported output format %d\n", (int)io->p.data_format);
        vip_unmap_buffer(io->buf);
        return -1;
    }
    vip_unmap_buffer(io->buf);
    return (int)count;
}
