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

// flashnet model structure (reused for feature length)
#define LEN_INPUT 12
#define LEN_LAYER_0 12
#define LEN_LAYER_M_1 128
#define LEN_LAYER_M_2 16
#define LEN_LAYER_0_HALF 6
#define LEN_LAYER_1 1

// # of devices
#define DEVICE_NUM 2

// # of history (we append previous 3 queue_len, latency and throughput)
#define N_HIST 3

// set to 1 to be verbose
#define VERBOSE 0

/*
 * Initialize surrogate decision tree state for a replay of total_io_num I/Os.
 * Allocates and initializes historical arrays for queue length, latency,
 * and throughput.
 */
void set_surrogate_dt(int total_io_num);

/*
 * Surrogate decision tree inference.
 *
 * Input:
 *   io_type: 1 = read, 0 = write
 *   size: IO size in bytes
 *   device: index of original device
 *   cur_queue_len: queue length when IO is submitted
 *
 * Output:
 *   0: accept IO on original device
 *   1: reject IO and redirect to secondary device
 */
int surrogate_dt_inference(long io_type, long size, uint32_t device, long cur_queue_len);

/*
 * Append the latest completed IO's metrics into the historical pools.
 */
void update_surrogate_dt(long io_queue_len, long io_latency, long io_throughput);

/*
 * Atomically increment and return current queue length.
 */
long add_fetch_cur_queue_len();

/*
 * Increment queue length by 1.
 */
void inc_queue_len();

/*
 * Decrement queue length by 1.
 */
void dec_queue_len();

/*
 * Free historical arrays allocated by set_surrogate_dt.
 */
void free_surrogate_dt();

#endif

