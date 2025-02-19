#include <iostream>

#include <chrono>
#include "gpm-helper.cuh"
#include "libgpmlog.cuh"

#include "gmemtable.h"
#include "debugger.h"
#include "gpu_batch.cuh"
#include "gpu_puts.cuh"
#include "helper.cuh"
#include "log.cuh"


#define NTHREADS_PER_BLK 1024
#define NBLKS 72

#define TIME_NOW std::chrono::high_resolution_clock::now()

GpuPuts::GpuPuts(GpuWriteBatch* batch, GMemtable** activeTable, GMemtableLog* gLog) : batch(batch), activeTable(activeTable), gLog(gLog) {
    keyLength = batch->keyLength;
    numWrites = (*activeTable)->size;
    Debugger debug(DEBUG); 
    allocateMemory((void**)&gIndices, numWrites * sizeof(uint64_t));
}


GpuPuts::~GpuPuts() {
    keyLength = 0;
    numWrites = 0;
}

__global__
void updateIndicesKernel(GpuWriteBatch* batch, GMemtable* table, GMemtableLog* gLog, uint64_t* dSortedIndices, uint64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= size) return;

    int keyLength = batch->keyLength;

    for(uint64_t i = idx; i < size; i += NTHREADS_PER_BLK * NBLKS) {
        int sorted_index = dSortedIndices[i]; 
        
        memcpy(table->keys + i * keyLength, batch->keys + sorted_index * keyLength, keyLength);
        cudaMemcpyAsync(gLog->keys + i * keyLength, batch->keys + sorted_index * keyLength, keyLength, cudaMemcpyDeviceToHost);
        
        table->valuePointer[i] = &batch->valuesPtr[sorted_index];
        gLog->valuePtrs[i] = &batch->valuesPtr[sorted_index];
        
        table->opID[i] = batch->opID[sorted_index];
        gLog->opID[i]= batch->opID[sorted_index];

        // printf("Key: %s\n", table->keys + i * keyLength);

        gpm_drain(); 
    }
}


//void sortPutsOnGPU(char *putKeys, char **putValuePointers, uint64_t *putOperationIDs, int numWrites, int keyLength, Memtable& table, int batchID) 
void GpuPuts::sortPutsOnGPU() {
    GMemtable* table = *activeTable; 
    allocateMemory((void**)&gIndices, numWrites * sizeof(uint64_t));
    cudaMemset(gIndices, 0, numWrites * sizeof(uint64_t));
    // char* keys = batch->keys; 
    
    debug.setDebugMode(DEBUG);
    // Sort the keys in the batch and insert them into the memtable
    // The sort functions give the indices for the sorted key list and not the sorted keys themselves

    if (numWrites < SORT_LIMIT) {
        sortSmaller(gIndices);
    } else {
        sortLarger(gIndices);
    }

    debug.print ("******************Sorted keys on GPU");
    // Rearrange the value pointers and operation IDs based on the sorted index array
    cudaError_t err = cudaPeekAtLastError();
    debug.print("Error: " + to_string(err)); 

    // The sorted indices are used to setup the memtable and WAL

    updateIndicesKernel<<<NBLKS, NTHREADS_PER_BLK>>>(batch, table, gLog, gIndices, numWrites); 
    cudaDeviceSynchronize();
    debug.print("********************Updated memtable and WAL with sorted keys");
    err = cudaPeekAtLastError();
    debug.print("Error: " + to_string(err)); 

    // Copy table->keys to CPU memory and print them
#if DEBUG
    char* cpuKeys = new char[keyLength * numWrites];
    cudaMemcpy(cpuKeys, table->keys, keyLength * numWrites, cudaMemcpyDeviceToHost);
    for (int i = 0; i < numWrites; i++) {
        std::cout << "[DEBUG]: " << "Key " << i << ": " << cpuKeys + i * keyLength << std::endl;
    }
    delete[] cpuKeys; 
#endif

}

