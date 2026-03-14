#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "atomic.h"
#include "dt_algo.h"

// DT headers generated from fixed_lat_dt training pipeline.
#include "dt_weights_header/w_Trace_dev_0_dt.h"
#include "dt_weights_header/w_Trace_dev_1_dt.h"

// FlashNet headers are used ONLY to calibrate latency budget.
#include "2ssds_weights_header/w_Trace_dev_0.h"
#include "2ssds_weights_header/w_Trace_dev_1.h"

long queue_len;
long hist_index;
long *prev_queue_len;
long *prev_latency;
long *prev_throughput;

int *dt_feature_all[DEVICE_NUM] = {dt_feature_dev_0, dt_feature_dev_1};
long *dt_threshold_all[DEVICE_NUM] = {dt_threshold_dev_0, dt_threshold_dev_1};
int *dt_left_all[DEVICE_NUM] = {dt_left_dev_0, dt_left_dev_1};
int *dt_right_all[DEVICE_NUM] = {dt_right_dev_0, dt_right_dev_1};
int *dt_value_all[DEVICE_NUM] = {dt_value_dev_0, dt_value_dev_1};
int dt_node_count[DEVICE_NUM] = {DT_DEV_0_NODE_COUNT, DT_DEV_1_NODE_COUNT};

// FlashNet weight views by device:
static long *shadow_devices_weights[][8] = {
    {weight_0_T_dev_0, weight_3_T_dev_0, bias_0_dev_0, bias_3_dev_0,
     weight_1_T_dev_0, bias_1_dev_0, weight_2_T_dev_0, bias_2_dev_0},
    {weight_0_T_dev_1, weight_3_T_dev_1, bias_0_dev_1, bias_3_dev_1,
     weight_1_T_dev_1, bias_1_dev_1, weight_2_T_dev_1, bias_2_dev_1},
};

// Per-device calibrated FlashNet inference budget (ns).
static uint64_t flashnet_budget_ns[DEVICE_NUM] = {0, 0};

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
}

static void wait_ns(uint64_t ns_to_wait) {
    if (ns_to_wait == 0) {
        return;
    }

    // For larger waits, yield the CPU first.
    if (ns_to_wait >= 100000ULL) {  // >= 100 us
        struct timespec ts;
        ts.tv_sec = ns_to_wait / 1000000000ULL;
        ts.tv_nsec = ns_to_wait % 1000000000ULL;
        nanosleep(&ts, NULL);
        return;
    }

    // For very short waits, busy-spin to avoid scheduler oversleep.
    uint64_t start = get_time_ns();
    while ((get_time_ns() - start) < ns_to_wait) {
    }
}

static int shadow_flashnet_forward_from_input(
    uint32_t device,
    const long input_vec_i[LEN_INPUT]
) {
    long *weights[8] = {
        shadow_devices_weights[device][0], shadow_devices_weights[device][1],
        shadow_devices_weights[device][2], shadow_devices_weights[device][3],
        shadow_devices_weights[device][4], shadow_devices_weights[device][5],
        shadow_devices_weights[device][6], shadow_devices_weights[device][7],
    };

    long mid_res_i[LEN_LAYER_0], mid_res_m_1[LEN_LAYER_M_1];
    long mid_res_m_2[LEN_LAYER_M_2], final_res_i[LEN_LAYER_1];
    long *weight_0_T_ent, *bias_0_ent, *weight_1_T_ent, *bias_1_ent;
    long *weight_M_1, *bias_M_1, *weight_M_2, *bias_M_2;

    weight_0_T_ent = weights[0];
    weight_1_T_ent = weights[1];
    bias_0_ent = weights[2];
    bias_1_ent = weights[3];
    weight_M_1 = weights[4];
    bias_M_1 = weights[5];
    weight_M_2 = weights[6];
    bias_M_2 = weights[7];

    for (int j = 0; j < LEN_LAYER_0; j++) {
        mid_res_i[j] = input_vec_i[j] - weight_0_T_ent[j];
        mid_res_i[j] = mid_res_i[j] * bias_0_ent[j];
    }

    for (int j = 0; j < LEN_LAYER_M_1; j++) {
        mid_res_m_1[j] = 0;
        for (int input_idx = 0; input_idx < LEN_LAYER_0; input_idx++) {
            mid_res_m_1[j] +=
                mid_res_i[input_idx] * weight_M_1[j * LEN_LAYER_0 + input_idx] >> 30;
        }
        mid_res_m_1[j] += bias_M_1[j];
        if (mid_res_m_1[j] < 0) {
            mid_res_m_1[j] = 0;
        }
    }

    for (int j = 0; j < LEN_LAYER_M_2; j++) {
        mid_res_m_2[j] = 0;
        for (int input_idx = 0; input_idx < LEN_LAYER_M_1; input_idx++) {
            mid_res_m_2[j] +=
                mid_res_m_1[input_idx] * weight_M_2[j * LEN_LAYER_M_1 + input_idx];
        }
        mid_res_m_2[j] += bias_M_2[j];
        if (mid_res_m_2[j] < 0) {
            mid_res_m_2[j] = 0;
        }
    }

    for (int j = 0; j < LEN_LAYER_1; j++) {
        final_res_i[j] = 0;
        for (int input_idx = 0; input_idx < LEN_LAYER_M_2; input_idx++) {
            final_res_i[j] +=
                mid_res_m_2[input_idx] * weight_1_T_ent[j * LEN_LAYER_M_2 + input_idx];
        }
        final_res_i[j] += bias_1_ent[j];
    }

    return (final_res_i[0] >= 0) ? 1 : 0;
}

static void calibrate_flashnet_budget(void) {
    // Deterministic synthetic input for latency calibration.
    long input_vec_i[LEN_INPUT] = {1, 4096, 1, 1, 1, 1, 100, 100, 100, 64, 64, 64};

    const int warmup_rounds = 200;
    const int sample_rounds = 4000;

    for (uint32_t dev = 0; dev < DEVICE_NUM; dev++) {
        for (int i = 0; i < warmup_rounds; i++) {
            (void)shadow_flashnet_forward_from_input(dev, input_vec_i);
        }

        uint64_t t0 = get_time_ns();
        for (int i = 0; i < sample_rounds; i++) {
            (void)shadow_flashnet_forward_from_input(dev, input_vec_i);
        }
        uint64_t t1 = get_time_ns();

        uint64_t avg_ns = (t1 - t0) / (uint64_t)sample_rounds;
        // Floor to 1 ns just to avoid zero-budget anomalies.
        flashnet_budget_ns[dev] = (avg_ns == 0) ? 1 : avg_ns;
    }

    printf(
        "[fixed_lat_dt] calibrated FlashNet inference budget (ns): dev0=%lu dev1=%lu\n",
        (unsigned long)flashnet_budget_ns[0],
        (unsigned long)flashnet_budget_ns[1]
    );
}

long add_fetch_cur_queue_len() { return atomic_inc_fetch(&queue_len); }
void inc_queue_len() { atomic_inc(&queue_len); }
void dec_queue_len() { atomic_dec(&queue_len); }

void set_surrogate_dt(int total_io_num) {
    queue_len = 0;
    hist_index = 0;

    prev_queue_len = malloc(total_io_num * sizeof(long));
    prev_latency = malloc(total_io_num * sizeof(long));
    prev_throughput = malloc(total_io_num * sizeof(long));

    if (prev_queue_len == NULL || prev_latency == NULL || prev_throughput == NULL) {
        printf("[Error] malloc failed in set_surrogate_dt\n");
        exit(1);
    }

    for (long i = 0; i < total_io_num; i++) {
        prev_queue_len[i] = -1;
        prev_latency[i] = -1;
        prev_throughput[i] = -1;
    }

    calibrate_flashnet_budget();
}

int surrogate_dt_inference(long io_type, long size, uint32_t device, long cur_queue_len) {
    long input_vec_i[LEN_INPUT];
    for (int i = 0; i < LEN_INPUT; i++) {
        input_vec_i[i] = 0;
    }
    input_vec_i[0] = io_type;
    input_vec_i[1] = size;
    input_vec_i[2] = cur_queue_len;

    long cur_hist_index = hist_index;
    for (int i = 1; i <= N_HIST; i++) {
        if (cur_hist_index - i >= 0) {
            input_vec_i[2 + i] = prev_queue_len[cur_hist_index - i];
            input_vec_i[5 + i] = prev_latency[cur_hist_index - i];
            input_vec_i[8 + i] = prev_throughput[cur_hist_index - i];

            if (prev_queue_len[cur_hist_index - i] == -1 ||
                prev_latency[cur_hist_index - i] == -1 ||
                prev_throughput[cur_hist_index - i] == -1) {
                printf("[Error] The historical data is not valid!");
                exit(1);
            }
        }
    }

    if (device >= DEVICE_NUM) {
        printf("[Error] device index %u out of range (DEVICE_NUM=%d)\n", device, DEVICE_NUM);
        exit(1);
    }

    uint64_t dt_start_ns = get_time_ns();

    int *feature = dt_feature_all[device];
    long *threshold = dt_threshold_all[device];
    int *left = dt_left_all[device];
    int *right = dt_right_all[device];
    int *value = dt_value_all[device];
    int node_count = dt_node_count[device];

    int pred = 0;
    int node = 0;
    while (1) {
        if (node < 0 || node >= node_count) {
            printf("[Error] invalid node index %d (node_count=%d)\n", node, node_count);
            exit(1);
        }

        int feat = feature[node];
        if (feat < 0) {
            pred = value[node];
            break;
        }
        if (feat >= LEN_INPUT) {
            printf("[Error] feature index %d out of range (LEN_INPUT=%d)\n", feat, LEN_INPUT);
            exit(1);
        }

        long feat_val = input_vec_i[feat];
        long thr = threshold[node];
        node = (feat_val <= thr) ? left[node] : right[node];
    }

    uint64_t dt_elapsed_ns = get_time_ns() - dt_start_ns;
    uint64_t budget_ns = flashnet_budget_ns[device];
    if (dt_elapsed_ns < budget_ns) {
        wait_ns(budget_ns - dt_elapsed_ns);
    }

    if (VERBOSE) {
        printf(
            "[fixed_lat_dt][dev %u] hist=%ld pred=%d dt_ns=%lu budget_ns=%lu\n",
            device,
            cur_hist_index,
            pred,
            (unsigned long)dt_elapsed_ns,
            (unsigned long)budget_ns
        );
    }

    return pred;
}

void update_surrogate_dt(long io_queue_len, long io_latency, long io_throughput) {
    prev_queue_len[hist_index] = io_queue_len;
    prev_latency[hist_index] = io_latency;
    prev_throughput[hist_index] = io_throughput;
    hist_index += 1;
}

void free_surrogate_dt() {
    free(prev_queue_len);
    free(prev_latency);
    free(prev_throughput);
}

