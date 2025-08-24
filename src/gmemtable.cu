#include <cuda_runtime.h>
#include "gmemtable.cuh"

static inline void cuda_free_if(void*& p) {
    if (!p) return;
    // Optional, but helps avoid freeing in-use buffers:
    cudaDeviceSynchronize();
    cudaFree(p);   // these were allocated via allocateMemory() → device/UM
    p = nullptr;
}

void GMemtable::freeGMemtable() {
    cuda_free_if(reinterpret_cast<void*&>(keys));
    cuda_free_if(reinterpret_cast<void*&>(valuePointer)); // array of device pointers-to-host
    cuda_free_if(reinterpret_cast<void*&>(opID));
    batchID = -1;
    keyLength = valueLength = 0;
    numKeys = size = 0;
    memtableID = -1;
    numImmutableMemtables = 0;
}

// If you also have a helper to free the struct itself (allocated with allocateMemoryManaged):
static inline void deleteMemtable(GMemtable*& mt) {
    if (!mt) return;
    mt->freeGMemtable();
    cudaFree(mt);   // allocateMemoryManaged → cudaMallocManaged
    mt = nullptr;
}

void CMemtable::freeCMemtable() {
    // Allocated with new[] in SstWriter::copyGMemtToCPU → delete[]
    delete[] keys;         keys = nullptr;
    delete[] valuePointer; valuePointer = nullptr;
    delete[] opID;         opID = nullptr;

    batchID = -1;
    keyLength = valueLength = 0;
    numKeys = size = 0;
    memtableID = -1;
}