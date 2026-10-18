#ifndef SURROGATE_LR_ALGO_H
#define SURROGATE_LR_ALGO_H

#include <stdint.h>

// Logistic-regression surrogate of FlashNet.
//
// Feature vector layout (matches FlashNet and the surrogate decision trees):
// 0: io_type
// 1: size
// 2: cur_queue_len
// 3-5: prev_queue_len_1..3
// 6-8: prev_latency_1..3
// 9-11: prev_throughput_1..3
//
// The model is one dot product on the raw (long) feature values:
//   z = lr_bias + sum_j lr_weight[j] * x[j]
//   reject (1) if z > 0, accept (0) otherwise.
// The training script folds its StandardScaler into lr_weight / lr_bias, so no
// normalization happens here (see train_surrogate_lr_fidelity.py).

#define LEN_INPUT 12
#define DEVICE_NUM 2
#define N_HIST 3
#define VERBOSE 0

void set_surrogate_lr(int total_io_num);
int surrogate_lr_inference(long io_type, long size, uint32_t device, long cur_queue_len);
void update_surrogate_lr(long io_queue_len, long io_latency, long io_throughput);
long add_fetch_cur_queue_len();
void inc_queue_len();
void dec_queue_len();
void free_surrogate_lr();

#endif
