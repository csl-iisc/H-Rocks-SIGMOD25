#include "debugger.h"
#include <iostream>
#include "gmemtable.h"
#include "gpu_updates.cuh"

bool string_compare(const char* a, const char* b, size_t length); 


struct string_comparator {
    const char* data;
    int keyLength;

    string_comparator(const char* _data, int _keyLength) : data(_data), keyLength(_keyLength) {}

    __device__ bool operator()(const int& a, const int& b) const {
        const char* keyA = data + a * keyLength;
        const char* keyB = data + b * keyLength;

        for (int i = 0; i < keyLength; ++i) {
            if (keyA[i] != keyB[i])
                return keyA[i] < keyB[i];
        }
        return false; // Return false if they are equal
    }
};


void GpuUpdates::sortSmallerUpdates(uint64_t* gIndices) {

    int keyLength = (*activeTable)->keyLength;
    uint64_t numUpdates = (*activeTable)->numKeys; 
    debug.print("numUpdates: " + std::to_string(numUpdates) + " KeyLength: " + std::to_string(keyLength));
    thrust::device_vector<uint64_t> indices(numUpdates);

    char* keys = batch->keys; 
    thrust::device_vector<char> dKeys(numUpdates * keyLength);
    cudaMemcpyAsync(thrust::raw_pointer_cast(dKeys.data()), keys, numUpdates * keyLength * sizeof(char), cudaMemcpyDeviceToDevice);
    // thrust::device_vector<int> indices(numUpdates);
    thrust::sequence(indices.begin(), indices.end());

    cudaError_t err = cudaPeekAtLastError();
    debug.print("Memory operations and setup error: " + std::to_string(err));

    // Sort using custom comparator
    thrust::sort(
        thrust::device,
        indices.begin(),
        indices.end(),
        string_comparator(thrust::raw_pointer_cast(dKeys.data()), keyLength)
    );

    err = cudaPeekAtLastError();
    debug.print("Thrust sort Error: " + std::to_string(err));

    cudaMemcpy(gIndices, thrust::raw_pointer_cast(indices.data()), numUpdates * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    
    err = cudaPeekAtLastError();
    debug.print("Memcpy sort Error: " + std::to_string(err));


#if 0
    char* sorted_keys = new char[numUpdates * keyLength];
    // copy the indices
    uint64_t* hIndices = new uint64_t[numUpdates];
    cudaMemcpy(hIndices, gIndices, numUpdates * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(sorted_keys, thrust::raw_pointer_cast(dKeys.data()), numUpdates * keyLength * sizeof(char), cudaMemcpyDeviceToHost);
    std::cout << "Sorted Keys:\n";
    for (int i = 0; i < numUpdates; i++) {
        std::cout << "i: " << i << " sorted index: " << hIndices[i] << " " << &sorted_keys[hIndices[i] * keyLength] << "\n";
    }
    delete[] sorted_keys;
#endif

}