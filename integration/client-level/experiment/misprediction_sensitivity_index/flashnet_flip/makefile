# Misprediction injection parameters (override: make FLIP_PROB=0.05 FLIP_SEED=7)
FLIP_PROB ?= 0.0
FLIP_SEED ?= 42

all:
	gcc -DFLIP_PROB=$(FLIP_PROB) -DFLIP_SEED=$(FLIP_SEED) flashnet_algo.c io_replayer.c -o io_replayer -lpthread
	echo "Done compiling (FLIP_PROB=$(FLIP_PROB), FLIP_SEED=$(FLIP_SEED))."
