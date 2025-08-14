#pragma once 
#include <iostream>


struct GpuWriteBatch {
    char* keys; 
    char* valuesPtr;
    uint64_t* opID;
    char* cKeys;
    uint64_t numWrites; 
    int keyLength; 
    int valueLength; 
    int batchID; 
};

typedef struct GpuWriteBatch GpuWriteBatch;

struct GpuReadBatch {
    char* keys;
    uint64_t* opID;
    char** outputValuesPtr;
    uint64_t numReads; 
    int keyLength; 
    int valueLength; 
    int batchID; 
}; 

typedef struct GpuReadBatch GpuReadBatch;

struct GpuRangeBatch {
    char* starKeys; 
    char* endKeys; 
    uint64_t* opID; 
    char* outputValuesPtr; 
    uint64_t numRangeQueries; 
    int keyLength; 
    int valueLength; 
    int batchID; 
};

typedef struct GpuRangeBatch GpuRangeBatch;

struct GpuUpdateBatch {
    char* keys; 
    uint64_t* opID; 
    char* valuesPtr;
    uint64_t numWrites; 
    int keyLength; 
    int valueLength; 
    int batchID;  
    char* cKeys; // For memory management   
    unsigned long long int numUpdates; // Number of updates in the batch
}; 

typedef struct GpuUpdateBatch GpuUpdateBatch;