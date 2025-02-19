#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include "gmemtable.h"
#include "libgpm.cuh"
#include "gpm-helper.cuh"
#include "libgpmlog.cuh"

struct log_entry_t {
    int index;
    char* key;
    char* value;
    uint64_t sequenceId;
};

__global__ void checkLogValid(gpmlog *dlog, bool *valid) {
    int partition_size = gpmlog_get_size(dlog, 0);
    *valid = (partition_size > 0);
}

__global__ void recoverFromLog(char *keys, char **values, uint64_t* seqID, gpmlog *dlog, bool *valid, int partitions, int keyLen) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= partitions) return;

    if (gpmlog_get_size(dlog, id) < sizeof(log_entry_t)) return;

    log_entry_t entry;
    gpmlog_read(dlog, &entry, sizeof(log_entry_t), id);
    memcpy(keys + entry.index * keyLen, entry.key, keyLen); // Assuming fixed length keys
    values[entry.index] = entry.value;
    seqID[entry.index] = entry.sequenceId;
}

void recoverGpmFiles(std::string folderName, GMemtable* gMemt) {
    // Initializations from the GMemtableLog setup
    uint64_t numKeys = gMemt->size;
    int keyLen = gMemt->keyLength;
    gpmlog *dlog = gpmlog_open("rdb_log"); // Open your log file
    int partitions = gpmlog_get_partitions(dlog);

    char* keys;
    char** values;
    uint64_t *seqId;

    cudaMalloc((void**)&keys, numKeys * keyLen);
    cudaMalloc((void**)&values, numKeys * sizeof(char*));
    cudaMalloc((void**)&seqId, numKeys * sizeof(uint64_t));

    bool *valid;
    cudaMallocHost((void**)&valid, sizeof(bool));

    checkLogValid<<<1, 1>>>(dlog, valid);
    cudaDeviceSynchronize();

    if (*valid) {
        dim3 blocks((partitions + 1024 - 1) / 1024);
        dim3 threads(1024);
        recoverFromLog<<<blocks, threads>>>(keys, values, seqId, dlog, valid, partitions, keyLen);
        cudaDeviceSynchronize();
    }

    gpmlog_close(dlog);
    cudaFree(keys);
    cudaFree(values);
    cudaFree(seqId);
    cudaFreeHost(valid);
}

int main() {
    GMemtable gMemt;
    std::string folderName = "/pmem/test_puts";
    recoverGpmFiles(folderName, &gMemt);
    return 0;
}
