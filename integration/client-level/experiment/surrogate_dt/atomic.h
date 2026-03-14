#ifndef SURROGATE_DT_ATOMIC_H
#define SURROGATE_DT_ATOMIC_H

// Minimal atomic helpers needed by io_replayer.c and dt_algo.c

#define atomic_read(ptr) __atomic_load_n((ptr), __ATOMIC_SEQ_CST)
#define atomic_fetch_inc(ptr) __atomic_fetch_add((ptr), 1, __ATOMIC_SEQ_CST)
#define atomic_inc_fetch(ptr) __atomic_add_fetch((ptr), 1, __ATOMIC_SEQ_CST)
#define atomic_inc(ptr) ((void)__atomic_fetch_add((ptr), 1, __ATOMIC_SEQ_CST))
#define atomic_dec(ptr) ((void)__atomic_fetch_sub((ptr), 1, __ATOMIC_SEQ_CST))
#define atomic_add(ptr, n) ((void)__atomic_fetch_add((ptr), (n), __ATOMIC_SEQ_CST))

#endif

