#include "string.cuh"

__device__ void string_copy(const char* src, char* dst, const int length)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        dst[tid] = src[tid];
    }
}

__device__ void stringCpy(char* dst, const char* src, const int length)
{
    for(int i = 0; i < length; ++i) {
        dst[i] = src[i];
    }
}

bool string_compare_bool(const char* a, const char* b, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (a[i] < b[i]) return false;
        if (a[i] > b[i]) return false;
    }
    return true;
}
