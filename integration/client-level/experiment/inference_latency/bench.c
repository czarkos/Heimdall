/*
    Inference latency / CPU cost microbenchmark for the client-level models.

    Links the exact inference code used by the replayers (flashnet/flashnet_algo.c or
    surrogate_dt/dt_algo.c) with the trained per-trace headers, and calls the model on real
    feature vectors taken from that trace's mldrive<dev>.csv (the history arrays are filled
    exactly as the replayer fills them, so the model reads the same 12 features).

    Build (done by run_bench.py):
        gcc -O0 -DMODEL_FLASHNET bench.c flashnet_algo.c -o bench
        gcc -O0 -DMODEL_DT       bench.c dt_algo.c       -o bench

    Usage:  ./bench <inputs.bin> <device 0|1> <cpu to pin> <timed passes>

    inputs.bin: int64 n, then 6 int64 arrays of length n:
        io_type, size, queue_len, hist_queue_len, hist_latency, hist_throughput
    where hist_*[k] is the (queue_len, latency, throughput) of IO k, i.e. what the replayer's
    history pool holds after IO k completes. Inference for IO r (a read, r >= 3) runs with
    hist_index = r, so it reads hist_*[r-1], hist_*[r-2], hist_*[r-3].

    Prints one JSON line with the raw measurements.
*/
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

#if defined(MODEL_FLASHNET)
#include "flashnet_algo.h"
#define MODEL_NAME "flashnet"
#define MODEL_INIT(n) set_flashnet(n)
#define MODEL_INFER(t, s, d, q) flashnet_inference(t, s, d, q)
#elif defined(MODEL_DT)
#include "dt_algo.h"
#define MODEL_NAME "dt"
#define MODEL_INIT(n) set_surrogate_dt(n)
#define MODEL_INFER(t, s, d, q) surrogate_dt_inference(t, s, d, q)
#else
#error "Define MODEL_FLASHNET or MODEL_DT"
#endif

/* Globals owned by the model's algo .c (shared history pool) */
extern long hist_index;
extern long *prev_queue_len;
extern long *prev_latency;
extern long *prev_throughput;

static volatile long sink;

/* Same signature as the models, used to measure loop + call overhead */
__attribute__((noinline)) int null_inference(long io_type, long size, uint32_t device, long cur_queue_len) {
    __asm__ volatile("" ::: "memory");
    return (int)((io_type + size + device + cur_queue_len) & 1);
}

static uint64_t now_ns(clockid_t clk) {
    struct timespec ts;
    clock_gettime(clk, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double pct(const uint64_t *sorted, long n, double p) {
    long i = (long)(p / 100.0 * (double)(n - 1) + 0.5);
    return (double)sorted[i < 0 ? 0 : (i >= n ? n - 1 : i)];
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <inputs.bin> <device> <cpu> <passes>\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    uint32_t device = (uint32_t)atoi(argv[2]);
    int cpu = atoi(argv[3]);
    int passes = atoi(argv[4]);

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
        return 1;
    }

    /* ---- load inputs ---- */
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return 1; }
    int64_t n;
    if (fread(&n, sizeof(n), 1, f) != 1) { fprintf(stderr, "bad input\n"); return 1; }
    int64_t *cols[6];
    for (int c = 0; c < 6; c++) {
        cols[c] = malloc(n * sizeof(int64_t));
        if (fread(cols[c], sizeof(int64_t), n, f) != (size_t)n) { fprintf(stderr, "bad input\n"); return 1; }
    }
    fclose(f);
    int64_t *io_type = cols[0], *size = cols[1], *qlen = cols[2];

    MODEL_INIT((int)n);
    for (long k = 0; k < n; k++) {
        prev_queue_len[k] = cols[3][k];
        prev_latency[k] = cols[4][k];
        prev_throughput[k] = cols[5][k];
    }

    /* Inference is only called for reads (as in the replayer) */
    long *idx = malloc(n * sizeof(long));
    long m = 0;
    for (long r = 3; r < n; r++) {
        if (io_type[r] == 1) idx[m++] = r;
    }
    if (m == 0) { fprintf(stderr, "no read IOs in input\n"); return 1; }

    /* ---- TSC calibration (constant_tsc: ticks per ns of wall time) ---- */
    uint64_t t0 = now_ns(CLOCK_MONOTONIC_RAW), c0 = __rdtsc();
    while (now_ns(CLOCK_MONOTONIC_RAW) - t0 < 200000000ULL) { }
    double ticks_per_ns = (double)(__rdtsc() - c0) / (double)(now_ns(CLOCK_MONOTONIC_RAW) - t0);

    /* ---- warm-up: >= 1 s of real work so the CPU frequency has ramped up ---- */
    long rejects = 0;
    uint64_t warm_start = now_ns(CLOCK_MONOTONIC_RAW);
    int warm_passes = 0;
    do {
        rejects = 0;
        for (long i = 0; i < m; i++) {
            hist_index = idx[i];
            rejects += MODEL_INFER(io_type[idx[i]], size[idx[i]], device, qlen[idx[i]]);
        }
        warm_passes++;
    } while (now_ns(CLOCK_MONOTONIC_RAW) - warm_start < 1000000000ULL || warm_passes < 2);
    sink += rejects;

    /* ---- batch passes: wall time, thread CPU time and TSC ticks per inference ---- */
    double *wall = malloc(passes * sizeof(double)), *cpu_t = malloc(passes * sizeof(double));
    double *ticks = malloc(passes * sizeof(double)), *null_wall = malloc(passes * sizeof(double));
    for (int p = 0; p < passes; p++) {
        uint64_t w0 = now_ns(CLOCK_MONOTONIC_RAW), u0 = now_ns(CLOCK_THREAD_CPUTIME_ID), k0 = __rdtsc();
        long acc = 0;
        for (long i = 0; i < m; i++) {
            hist_index = idx[i];
            acc += MODEL_INFER(io_type[idx[i]], size[idx[i]], device, qlen[idx[i]]);
        }
        uint64_t k1 = __rdtsc(), u1 = now_ns(CLOCK_THREAD_CPUTIME_ID), w1 = now_ns(CLOCK_MONOTONIC_RAW);
        sink += acc;
        wall[p] = (double)(w1 - w0) / m;
        cpu_t[p] = (double)(u1 - u0) / m;
        ticks[p] = (double)(k1 - k0) / m;

        /* same loop with an empty function: loop + call overhead */
        w0 = now_ns(CLOCK_MONOTONIC_RAW);
        acc = 0;
        for (long i = 0; i < m; i++) {
            hist_index = idx[i];
            acc += null_inference(io_type[idx[i]], size[idx[i]], device, qlen[idx[i]]);
        }
        w1 = now_ns(CLOCK_MONOTONIC_RAW);
        sink += acc;
        null_wall[p] = (double)(w1 - w0) / m;
    }

    /* ---- per-call distribution (serialized rdtsc around each call) ---- */
    uint64_t *lat = malloc(m * sizeof(uint64_t)), *ovh = malloc(m * sizeof(uint64_t));
    for (long i = 0; i < m; i++) {
        hist_index = idx[i];
        _mm_lfence();
        uint64_t a = __rdtsc();
        _mm_lfence();
        int res = MODEL_INFER(io_type[idx[i]], size[idx[i]], device, qlen[idx[i]]);
        _mm_lfence();
        uint64_t b = __rdtsc();
        _mm_lfence();
        sink += res;
        lat[i] = b - a;
    }
    for (long i = 0; i < m; i++) {   /* timer overhead: same fences, no call */
        _mm_lfence();
        uint64_t a = __rdtsc();
        _mm_lfence();
        _mm_lfence();
        uint64_t b = __rdtsc();
        _mm_lfence();
        ovh[i] = b - a;
    }
    qsort(lat, m, sizeof(uint64_t), cmp_u64);
    qsort(ovh, m, sizeof(uint64_t), cmp_u64);
    double ovh_med = pct(ovh, m, 50);
    double pc[5];
    double ps[5] = {50, 90, 99, 99.9, 100};
    for (int i = 0; i < 5; i++) {
        double v = pct(lat, m, ps[i]) - ovh_med;
        pc[i] = (v < 0 ? 0 : v) / ticks_per_ns;
    }

    /* ---- summarize batch passes ---- */
    double w_mean = 0, c_mean = 0, k_mean = 0, n_mean = 0;
    for (int p = 0; p < passes; p++) { w_mean += wall[p]; c_mean += cpu_t[p]; k_mean += ticks[p]; n_mean += null_wall[p]; }
    w_mean /= passes; c_mean /= passes; k_mean /= passes; n_mean /= passes;
    double w_var = 0;
    for (int p = 0; p < passes; p++) w_var += (wall[p] - w_mean) * (wall[p] - w_mean);
    double w_std = passes > 1 ? __builtin_sqrt(w_var / (passes - 1)) : 0;
    qsort(wall, passes, sizeof(double), cmp_double);

    printf("{\"model\": \"%s\", \"device\": %u, \"cpu\": %d, \"n_rows\": %ld, \"n_inferences\": %ld, "
           "\"passes\": %d, \"warmup_passes\": %d, \"ticks_per_ns\": %.4f, "
           "\"wall_ns_mean\": %.3f, \"wall_ns_median\": %.3f, \"wall_ns_std\": %.3f, "
           "\"cpu_ns_mean\": %.3f, \"tsc_ticks_mean\": %.3f, \"loop_overhead_ns\": %.3f, "
           "\"percall_ns_p50\": %.2f, \"percall_ns_p90\": %.2f, \"percall_ns_p99\": %.2f, "
           "\"percall_ns_p999\": %.2f, \"percall_ns_max\": %.2f, \"timer_overhead_ticks\": %.1f, "
           "\"reject_rate\": %.5f}\n",
           MODEL_NAME, device, cpu, (long)n, m, passes, warm_passes, ticks_per_ns,
           w_mean, wall[passes / 2], w_std, c_mean, k_mean, n_mean,
           pc[0], pc[1], pc[2], pc[3], pc[4], ovh_med, (double)rejects / m);
    return 0;
}
