#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/fs.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>

#include "atomic.h"
#include "dt_algo.h"

enum {
    READ_IO = 1,
    WRITE_IO = 0,
};

FILE *out_file;
int nr_workers = 8;
int64_t jobtracker = 0;
int block_size = 1;
char **device_list = NULL;
int total_dev_num = 0;
char device_list_str[600];
int original_device_index = -100;
char tracefile[600];
char output_file[600];
int duration = 0;
int *fds = NULL;
int64_t *DISKSZ = NULL;

int64_t nr_tt_ios;
int64_t latecount = 0;
int64_t slackcount = 0;
uint64_t starttime;
void *buff;
int respecttime = 1;

int64_t *oft;
int *reqsize;
int *reqflag;
float *timestamp;
int64_t progress;

long *req_queue_len;
long *req_latency;
long *req_throughput;
enum {
    IO_NOT_COMPLETED = -1,
    IO_REJECTED = -2,
    IO_HIST_APPENDED = -3,
};

pthread_mutex_t lock;
static int QuitFlag = 0;
static pthread_mutex_t QuitMutex = PTHREAD_MUTEX_INITIALIZER;

void setQuitFlag(void) {
    pthread_mutex_lock(&QuitMutex);
    QuitFlag = 1;
    pthread_mutex_unlock(&QuitMutex);
}

int shouldQuit(void) {
    int temp;
    pthread_mutex_lock(&QuitMutex);
    temp = QuitFlag;
    pthread_mutex_unlock(&QuitMutex);
    return temp;
}

static int64_t get_disksz(int devfd) {
    int64_t sz;
    ioctl(devfd, BLKGETSIZE64, &sz);
    return sz;
}

int64_t read_trace(char ***req, char *tracefile) {
    char line[1024];
    int64_t nr_lines = 0, i = 0;
    int ch;

    FILE *trace = fopen(tracefile, "r");
    if (trace == NULL) {
        printf("Cannot open trace file: %s!\n", tracefile);
        exit(1);
    }

    while (!feof(trace)) {
        ch = fgetc(trace);
        if (ch == '\n') nr_lines++;
    }
    rewind(trace);

    if ((*req = malloc(nr_lines * sizeof(char *))) == NULL) {
        fprintf(stderr, "memory allocation error (%d)!\n", __LINE__);
        exit(1);
    }

    while (fgets(line, sizeof(line), trace) != NULL) {
        line[strlen(line) - 1] = '\0';
        if (((*req)[i] = malloc((strlen(line) + 1) * sizeof(char))) == NULL) {
            fprintf(stderr, "memory allocation error (%d)!\n", __LINE__);
            exit(1);
        }
        strcpy((*req)[i], line);
        i++;
    }
    fclose(trace);
    return nr_lines;
}

void parse_io(char **reqs, int total_io) {
    char *one_io;
    int64_t i = 0;

    oft = malloc(total_io * sizeof(int64_t));
    reqsize = malloc(total_io * sizeof(int));
    reqflag = malloc(total_io * sizeof(int));
    timestamp = malloc(total_io * sizeof(float));
    req_queue_len = malloc(total_io * sizeof(long));
    req_latency = malloc(total_io * sizeof(long));
    req_throughput = malloc(total_io * sizeof(long));

    if (!oft || !reqsize || !reqflag || !timestamp || !req_queue_len || !req_latency || !req_throughput) {
        printf("memory allocation error (%d)!\n", __LINE__);
        exit(1);
    }

    one_io = malloc(1024);
    if (one_io == NULL) {
        fprintf(stderr, "memory allocation error (%d)!\n", __LINE__);
        exit(1);
    }
    for (i = 0; i < total_io; i++) {
        memset(one_io, 0, 1024);
        strcpy(one_io, reqs[i]);

        timestamp[i] = atof(strtok(one_io, " "));
        strtok(NULL, " ");
        oft[i] = atoll(strtok(NULL, " "));
        reqsize[i] = atoi(strtok(NULL, " ")) * block_size;
        reqflag[i] = atoi(strtok(NULL, " "));

        req_queue_len[i] = IO_NOT_COMPLETED;
        req_latency[i] = IO_NOT_COMPLETED;
        req_throughput[i] = IO_NOT_COMPLETED;
    }

    free(one_io);
}

int mkdirr(const char *path, const mode_t mode, const int fail_on_exist) {
    int result = 0;
    char *dir = NULL;
    do {
        if ((dir = strrchr(path, '/'))) {
            *dir = '\0';
            result = mkdirr(path, mode, fail_on_exist);
            *dir = '/';
            if (result) break;
        }

        if (strlen(path)) {
            if ((result = mkdir(path, mode))) {
                result = 0;
            }
        }
    } while (0);
    return result;
}

void create_file(char *output_file) {
    if (-1 == mkdirr(output_file, 0755, 0)) {
        perror("mkdirr() failed()");
        exit(1);
    }
    remove(output_file);
    out_file = fopen(output_file, "w");
    if (!out_file) {
        printf("Error creating out_file(%s) file!\n", output_file);
        exit(1);
    }
}

int surrogate_dt_algo(long io_type, long size, long cur_queue_len) {
    int target_device = original_device_index;
    int prediction_result = surrogate_dt_inference(io_type, size, target_device, cur_queue_len);
    if (io_type == READ_IO) {
        if (prediction_result == 1) {
            target_device = (target_device + 1) % total_dev_num;
            if (target_device == original_device_index) {
                printf("Error when changing target device!\n");
                exit(1);
            }
        }
    } else {
        printf("Surrogate decision tree only applied to read IO\n");
        exit(1);
    }
    return target_device;
}

void *perform_io() {
    int64_t cur_idx;
    int mylatecount = 0;
    int myslackcount = 0;
    struct timeval t1, t2;
    useconds_t sleep_time;
    int ret;

    while (!shouldQuit()) {
        cur_idx = atomic_fetch_inc(&jobtracker);
        if (cur_idx >= nr_tt_ios) {
            int64_t div = cur_idx / nr_tt_ios;
            cur_idx -= (nr_tt_ios * div);
            timestamp[cur_idx] += timestamp[nr_tt_ios - 1];
        }
        if (timestamp[cur_idx] > duration * 1000) break;

        myslackcount = 0;
        mylatecount = 0;

        if (respecttime == 1) {
            gettimeofday(&t1, NULL);
            int64_t elapsedtime = t1.tv_sec * 1e6 + t1.tv_usec - starttime;
            if (elapsedtime < (int64_t)(timestamp[cur_idx] * 1000)) {
                sleep_time = (useconds_t)(timestamp[cur_idx] * 1000) - elapsedtime;
                if (sleep_time > 100000) myslackcount++;
                usleep(sleep_time);
            } else {
                mylatecount++;
            }
        }

        gettimeofday(&t1, NULL);
        float submission_ts = (t1.tv_sec * 1e6 + t1.tv_usec - starttime) / 1000;

        long cur_queue_len = add_fetch_cur_queue_len();
        int target_device = original_device_index;
        if (reqflag[cur_idx] == READ_IO) {
            target_device = surrogate_dt_algo((long)reqflag[cur_idx], (long)reqsize[cur_idx], cur_queue_len);
        }

        if (target_device != original_device_index) {
            dec_queue_len();
        }

        if (reqflag[cur_idx] == WRITE_IO || reqflag[cur_idx] == READ_IO) {
            int64_t adjust_offset = oft[cur_idx];
            adjust_offset *= block_size;
            adjust_offset %= DISKSZ[target_device];
            adjust_offset = adjust_offset / 4096 * 4096;
            assert(adjust_offset >= 0);

            if (reqflag[cur_idx] == WRITE_IO) {
                ret = pwrite(fds[target_device], buff, reqsize[cur_idx], adjust_offset);
            } else {
                ret = pread(fds[target_device], buff, reqsize[cur_idx], adjust_offset);
            }
            if (ret < 0) {
                printf("IO error at index %ld, ret=%d, errno=%d\n", cur_idx, ret, errno);
            }
        } else {
            printf("Bad request type(%d)!\n", reqflag[cur_idx]);
            exit(1);
        }

        gettimeofday(&t2, NULL);
        int lat = (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_usec - t1.tv_usec);

        if (target_device == original_device_index) {
            req_queue_len[cur_idx] = cur_queue_len;
            req_latency[cur_idx] = lat;
            req_throughput[cur_idx] = reqsize[cur_idx] / (lat + 1);
            dec_queue_len();
        } else {
            req_queue_len[cur_idx] = IO_REJECTED;
            req_latency[cur_idx] = IO_REJECTED;
            req_throughput[cur_idx] = IO_REJECTED;
        }

        pthread_mutex_lock(&lock);
        fprintf(out_file, "%.3f,%d,%d,%d,%ld,%.3f,%d\n", timestamp[cur_idx], lat, reqflag[cur_idx], reqsize[cur_idx], oft[cur_idx], submission_ts, ret);
        pthread_mutex_unlock(&lock);

        atomic_add(&latecount, mylatecount);
        atomic_add(&slackcount, myslackcount);
    }
    return NULL;
}

void *pr_progress() {
    while (!shouldQuit()) {
        progress = atomic_read(&jobtracker);
        int64_t cur_late_cnt = atomic_read(&latecount);
        int64_t cur_slack_cnt = atomic_read(&slackcount);
        int64_t np = (progress > nr_tt_ios) ? progress : nr_tt_ios;
        printf(
            "Progress: %.2f%% (%lu/%lu), Late rate: %.2f%% (%lu), Slack rate: %.2f%% (%lu)\r",
            100 * (float)progress / np, progress, np,
            100 * (float)cur_late_cnt / progress, cur_late_cnt,
            100 * (float)cur_slack_cnt / progress, cur_slack_cnt
        );
        fflush(stdout);
        sleep(1);
    }
    printf("\n Finished replaying!\n");
    return NULL;
}

void *hist_update_thread() {
    long complete_idx = 0;
    while (!shouldQuit()) {
        long cur_idx = atomic_read(&jobtracker);
        for (int i = complete_idx; i <= cur_idx; i++) {
            if (req_queue_len[i] != IO_NOT_COMPLETED && req_latency[i] != IO_NOT_COMPLETED && req_throughput[i] != IO_NOT_COMPLETED) {
                if (i == complete_idx) complete_idx += 1;
                if (req_queue_len[i] >= 0 && req_latency[i] >= 0 && req_throughput[i] >= 0) {
                    update_surrogate_dt(req_queue_len[i], req_latency[i], req_throughput[i]);
                    req_queue_len[i] = IO_HIST_APPENDED;
                    req_latency[i] = IO_HIST_APPENDED;
                    req_throughput[i] = IO_HIST_APPENDED;
                }
            }
        }
    }
    return NULL;
}

void do_replay(void) {
    pthread_t track_thread;
    pthread_t update_thread;
    struct timeval t1, t2;
    int t;

    pthread_t *tid = malloc(nr_workers * sizeof(pthread_t));
    if (tid == NULL) {
        printf("Error malloc thread, LOC(%d)!\n", __LINE__);
        exit(1);
    }

    assert(pthread_mutex_init(&lock, NULL) == 0);

    gettimeofday(&t1, NULL);
    starttime = t1.tv_sec * 1000000 + t1.tv_usec;
    for (t = 0; t < nr_workers; t++) {
        assert(pthread_create(&tid[t], NULL, perform_io, NULL) == 0);
    }
    assert(pthread_create(&track_thread, NULL, pr_progress, NULL) == 0);
    assert(pthread_create(&update_thread, NULL, hist_update_thread, NULL) == 0);

    sleep(duration);
    setQuitFlag();
    for (t = 0; t < nr_workers; t++) pthread_join(tid[t], NULL);
    pthread_join(track_thread, NULL);
    pthread_join(update_thread, NULL);
    free(tid);

    gettimeofday(&t2, NULL);
    float totaltime = (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_usec - t1.tv_usec) / 1e3;
    float runtime = totaltime / 1000;
    float late_rate = 100 * (float)atomic_read(&latecount) / atomic_read(&jobtracker);
    float slack_rate = 100 * (float)atomic_read(&slackcount) / atomic_read(&jobtracker);

    fclose(out_file);
    assert(pthread_mutex_destroy(&lock) == 0);

    char command[1900];
    snprintf(command, sizeof(command), "%s %s %.2f %.2f %.2f %s %s.stats",
             "python statistics.py ", output_file, runtime, late_rate, slack_rate, " > ", output_file);
    system(command);
}

void free_mem() {
    free(oft);
    free(reqsize);
    free(reqflag);
    free(timestamp);
    free(req_queue_len);
    free(req_latency);
    free(req_throughput);
    free(DISKSZ);
    free(fds);
}

int main(int argc, char **argv) {
    char **request;

    if (argc != 6) {
        printf("Usage: ./io_replayer_dt $original_device_index $devices_list $trace $output_file $duration\n");
        exit(1);
    } else {
        original_device_index = atoi(argv[1]);
        sprintf(device_list_str, "%s", argv[2]);
        duration = atoi(argv[5]);
        char *dev_name = strtok(device_list_str, "-");
        while (dev_name != NULL) {
            device_list = realloc(device_list, (total_dev_num + 1) * sizeof(char *));
            device_list[total_dev_num] = dev_name;
            total_dev_num += 1;
            dev_name = strtok(NULL, "-");
        }
        sprintf(tracefile, "%s", argv[3]);
        sprintf(output_file, "%s", argv[4]);
    }

    fds = malloc(total_dev_num * sizeof(int));
    DISKSZ = malloc(total_dev_num * sizeof(int64_t));
    for (int i = 0; i < total_dev_num; i++) {
        int fd = open(device_list[i], O_DIRECT | O_RDWR);
        if (fd < 0) {
            printf("Cannot open %s\n", device_list[i]);
            exit(1);
        }
        fds[i] = fd;
        DISKSZ[i] = get_disksz(fd);
    }

    int total_io = read_trace(&request, tracefile);
    create_file(output_file);
    parse_io(request, total_io);

    int LARGEST_REQUEST_SIZE = (8 * 1024 * 1024);
    int MEM_ALIGN = 4096 * 8;
    if (posix_memalign(&buff, MEM_ALIGN, LARGEST_REQUEST_SIZE * block_size)) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }

    set_surrogate_dt(total_io);
    nr_tt_ios = total_io;
    do_replay();

    free(buff);
    free_surrogate_dt();
    free_mem();
    return 0;
}

