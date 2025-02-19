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

class GpuUpdates {
    private: 
        GpuUpdateBatch* batch; 
        GMemtable** activeTable;
        GMemtable** immutableTables;
        GMemtableLog* gLog; 
        int keyLength; 
        uint64_t numUpdates; 
        uint64_t* gIndices; 
        void sortSmallerUpdates(uint64_t* gIndices); 
        void sortLargerUpdates(uint64_t* gIndices); 
        Debugger debug; 

    public: 
        GpuUpdates(GpuUpdateBatch* batch, GMemtable** activeTable, GMemtable** immutableTable,GMemtableLog* gLog); 
        void updatesOnGpu(); 
        ~GpuUpdates(); 
}; 