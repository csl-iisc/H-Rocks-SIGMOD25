#pragma once
#include <stdio.h>

void allocateMemoryManaged(void** ptr, size_t size);
void allocateMemory(void** ptr, size_t size);
void allocateMemoryHost(void** ptr, size_t size);
void freeMemory(void* ptr);
void copyMemory(void* dst, void* src, size_t size, cudaMemcpyKind kind);
void copyMemoryAsync(void* dst, void* src, size_t size, cudaMemcpyKind kind);
void synchronizeDevice();
void setDevice(int device);
void getDeviceCount(int* count);
void registerHost(void* ptr, size_t size);
void getDeviceProperties(cudaDeviceProp* prop, int device); 
void checkError(cudaError_t err);