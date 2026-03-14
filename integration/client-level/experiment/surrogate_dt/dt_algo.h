#ifndef SURROGATE_DT_ALGO_H
#define SURROGATE_DT_ALGO_H

#include <stdint.h>

// Feature vector layout (matches FlashNet):
// 0: io_type
// 1: size
// 2: cur_queue_len
// 3-5: prev_queue_len_1..3
// 6-8: prev_latency_1..3
// 9-11: prev_throughput_1..3

#define LEN_INPUT 12
#define DEVICE_NUM 2
#define N_HIST 3
#define VERBOSE 0

void set_surrogate_dt(int total_io_num);
int surrogate_dt_inference(long io_type, long size, uint32_t device, long cur_queue_len);
void update_surrogate_dt(long io_queue_len, long io_latency, long io_throughput);
long add_fetch_cur_queue_len();
void inc_queue_len();
void dec_queue_len();
void free_surrogate_dt();

#endif

