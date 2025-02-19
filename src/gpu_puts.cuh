#pragma once 

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <cuda_runtime_api.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include "debugger.h"
#include "gmemtable.h"
#include "gpu_batch.cuh"
#include "log.cuh"

#define SORT_LIMIT 100000000

class GpuPuts {
    private: 
        GpuWriteBatch* batch; 
        GMemtable** activeTable;
        GMemtableLog* gLog; 
        int keyLength; 
        uint64_t numWrites; 
        uint64_t* gIndices; 
        void sortSmaller(uint64_t* gIndices); 
        void sortLarger(uint64_t* gIndices); 
        Debugger debug; 

    public: 
        GpuPuts(GpuWriteBatch* batch, GMemtable** activeTable, GMemtableLog* gLog); 
        void sortPutsOnGPU(); 
        ~GpuPuts(); 
}; 