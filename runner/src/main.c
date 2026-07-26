/* main.c - CLI for the chained NPU FaceLandmarker runner.
 *
 *   fl_run --models ../compiled --ppm face.ppm            one frame, JSON out
 *   fl_run --models ../compiled --ppm face.ppm --loop 100 latency bench
 *
 * PPM (P6) keeps this tool dependency-free. Make one with ffmpeg:
 *   ffmpeg -i face.png -pix_fmt rgb24 face.ppm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "facelandmarker.h"

static unsigned char *read_ppm(const char *path, int *w, int *h)
{
    FILE *f = fopen(path, "rb");
    unsigned char *data;
    int maxv;
    char magic[3] = {0};
    if (!f) { perror(path); return NULL; }
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fprintf(stderr, "%s: not a P6 PPM\n", path); fclose(f); return NULL;
    }
    /* skip comments */
    int c;
    int vals[3] = {0, 0, 0}, n = 0;
    size_t bytes;
    while (n < 3 && (c = fgetc(f)) != EOF) {
        if (c == '#') { while ((c = fgetc(f)) != EOF && c != '\n'); }
        else if (c >= '0' && c <= '9') {
            int v = 0;
            do { v = v * 10 + (c - '0'); c = fgetc(f); } while (c >= '0' && c <= '9');
            vals[n++] = v;
        }
    }
    /* A header cut short leaves values unread: refuse the file rather than
     * size an allocation from whatever was on the stack. */
    if (n != 3) { fprintf(stderr, "%s: incomplete PPM header\n", path); fclose(f); return NULL; }
    *w = vals[0]; *h = vals[1]; maxv = vals[2];
    if (*w <= 0 || *h <= 0 || *w > 65535 || *h > 65535) {
        fprintf(stderr, "%s: implausible size %dx%d\n", path, *w, *h); fclose(f); return NULL;
    }
    if (maxv != 255) { fprintf(stderr, "%s: maxval %d unsupported\n", path, maxv); fclose(f); return NULL; }
    bytes = (size_t)*w * (size_t)*h * 3;
    data = (unsigned char *)malloc(bytes);
    if (!data) { fprintf(stderr, "%s: out of memory\n", path); fclose(f); return NULL; }
    if (fread(data, 1, bytes, f) != bytes) {
        fprintf(stderr, "%s: truncated\n", path); free(data); fclose(f); return NULL;
    }
    fclose(f);
    return data;
}

static int cmp_double(const void *a, const void *b)
{
    double d = *(const double *)a - *(const double *)b;
    return d < 0 ? -1 : d > 0 ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *models = "../compiled", *ppm = NULL;
    int loop = 1, i;
    double fps = 0.0;   /* >0: pace the loop at this rate (power measurements) */

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--models") && i + 1 < argc) models = argv[++i];
        else if (!strcmp(argv[i], "--ppm") && i + 1 < argc) ppm = argv[++i];
        else if (!strcmp(argv[i], "--loop") && i + 1 < argc) loop = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fps") && i + 1 < argc) fps = atof(argv[++i]);
        else {
            fprintf(stderr,
                "usage: %s --models DIR --ppm IMG.ppm [--loop N] [--fps F]\n", argv[0]);
            return 2;
        }
    }
    if (!ppm) { fprintf(stderr, "need --ppm (P6 rgb24). ffmpeg -i x.png -pix_fmt rgb24 x.ppm\n"); return 2; }

    int w, h;
    unsigned char *rgb = read_ppm(ppm, &w, &h);
    if (!rgb) return 1;
    fprintf(stderr, "frame %dx%d, models %s\n", w, h, models);

    /* FL_DETECTOR points at a face detector directory of its own, so the
     * full-range variant can be tried without touching the other models. */
    const char *det_dir = getenv("FL_DETECTOR");
    fl_ctx_t *ctx = fl_create_detector(models, det_dir);
    if (ctx && det_dir)
        fprintf(stderr, "using detector from %s\n", det_dir);
    if (!ctx) { free(rgb); return 1; }

    fl_result_t res;
    double *totals = (double *)malloc(sizeof(double) * (loop > 0 ? loop : 1));
    int runs = loop > 0 ? loop : 1;

    for (i = 0; i < runs; i++) {
        if (fl_process_rgb(ctx, rgb, w, h, &res) != 0) {
            fprintf(stderr, "fl_process_rgb failed at run %d\n", i);
            fl_destroy(ctx); free(rgb); free(totals);
            return 1;
        }
        totals[i] = res.ms_total;
        if (fps > 0.0) {
            double left_ms = 1000.0 / fps - res.ms_total;
            if (left_ms > 0) {
                struct timespec ts = { (time_t)(left_ms / 1000.0),
                                       (long)((left_ms - (long)(left_ms / 1000.0) * 1000.0) * 1e6) };
                nanosleep(&ts, NULL);
            }
        }
    }

    /* last-run JSON on stdout */
    printf("{\"face\": %s, \"detect_score\": %.4f, \"presence\": %.4f,\n",
           res.face_present ? "true" : "false", res.detect_score, res.presence_score);
    printf(" \"roi\": {\"cx\": %.1f, \"cy\": %.1f, \"size\": %.1f, \"rot_rad\": %.4f},\n",
           res.roi_cx, res.roi_cy, res.roi_size, res.roi_rotation);
    printf(" \"ms\": {\"detect\": %.2f, \"landmarks\": %.2f, \"blendshapes\": %.2f, \"total\": %.2f},\n",
           res.ms_detect, res.ms_landmarks, res.ms_blendshapes, res.ms_total);
    printf(" \"blendshapes\": {");
    for (i = 0; i < FL_NUM_BLENDSHAPES; i++)
        printf("%s\"%s\": %.4f", i ? ", " : "", FL_BLENDSHAPE_NAMES[i], res.blendshapes[i]);
    printf("},\n \"landmarks_px\": [");
    for (i = 0; i < FL_NUM_LANDMARKS; i++)
        printf("%s[%.1f,%.1f,%.1f]", i ? "," : "",
               res.landmarks[i].x, res.landmarks[i].y, res.landmarks[i].z);
    printf("]}\n");

    if (runs > 1) {
        qsort(totals, runs, sizeof(double), cmp_double);
        fprintf(stderr,
            "chain latency over %d runs: p50=%.2f ms  p90=%.2f ms  p99=%.2f ms  (%.1f fps at p50)\n",
            runs, totals[runs / 2], totals[(int)(runs * 0.9)],
            totals[(int)(runs * 0.99)], 1000.0 / totals[runs / 2]);
    }

    fl_destroy(ctx);
    free(rgb);
    free(totals);
    return 0;
}
