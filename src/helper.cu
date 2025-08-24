#include "helper.cuh"
#include "debugger.h"
#include <string>

Debugger debug(DEBUG); 

void allocateMemoryManaged(void** ptr, size_t size) {
    cudaMallocManaged(ptr, size);
    debug.print("Allocated memory for pointer: " + std::string(typeid(*ptr).name()) + " with size: " + std::to_string(size) + " bytes");
    cudaError_t err = cudaPeekAtLastError();
    debug.print("Error: " + std::to_string(err)); 
}

void allocateMemory(void** ptr, size_t size) {
    cudaMalloc(ptr, size);    
    debug.print("Allocated memory for pointer: " + std::string(typeid(*ptr).name()) + " with size: " + std::to_string(size) + " bytes");
    cudaError_t err = cudaPeekAtLastError(); 
    debug.print("Error: " + std::to_string(err)); 
}

void freeMemory(void* ptr) {
    if (!ptr) {
        debug.print("Pointer is null, nothing to free.");
        return;
    }
    cudaFree(ptr);
    debug.print("Freed memory for pointer: " + std::string(typeid(ptr).name()));
    cudaError_t err = cudaPeekAtLastError(); 
    debug.print("Error: " + std::to_string(err)); 
}

void allocateMemoryHost(void** ptr, size_t size) {
    cudaMallocHost(ptr, size);
    debug.print("Allocated host memory for pointer: " + std::string(typeid(*ptr).name()) + " with size: " + std::to_string(size) + " bytes");
    cudaError_t err = cudaPeekAtLastError(); 
    debug.print("Error: " + std::to_string(err)); 
}

void copyMemory(void* dst, void* src, size_t size, cudaMemcpyKind kind) {
    cudaMemcpy(dst, src, size, kind);
    debug.print("Copied memory from " + std::string(typeid(src).name()) + " to " + std::string(typeid(dst).name()) + " with size: " + std::to_string(size) + " bytes");
    cudaError_t err = cudaPeekAtLastError(); 
    debug.print("Error: " + std::to_string(err)); 
}

void copyMemoryAsync(void* dst, void* src, size_t size, cudaMemcpyKind kind) {
    cudaMemcpyAsync(dst, src, size, kind);
    debug.print("Copied memory async from " + std::string(typeid(src).name()) + " to " + std::string(typeid(dst).name()) + " with size: " + std::to_string(size) + " bytes");
    cudaError_t err = cudaPeekAtLastError(); 
    debug.print("Error: " + std::to_string(err)); 
}

void synchronizeDevice() {
    cudaDeviceSynchronize();
    debug.print("Synchronized device");
    cudaError_t err = cudaPeekAtLastError(); 
    debug.print("Error: " + std::to_string(err)); 
}

void setDevice(int device) {
    cudaSetDevice(device);
    cudaError_t err = cudaPeekAtLastError(); 
    debug.print("Error: " + std::to_string(err)); 
}

void getDeviceCount(int* count) {
    cudaGetDeviceCount(count);
    cudaError_t err = cudaPeekAtLastError(); 
    debug.print("Error: " + std::to_string(err)); 
}

void registerHost(void* ptr, size_t size) {
    cudaHostRegister(ptr, size, 0);
    debug.print("Registered host memory for pointer: " + std::string(typeid(ptr).name()) + " with size: " + std::to_string(size) + " bytes");
    cudaError_t err = cudaPeekAtLastError(); 
    debug.print("Error: " + std::to_string(err)); 
}