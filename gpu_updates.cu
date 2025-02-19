#include <iostream>

#include <chrono>
#include "gpm-helper.cuh"
#include "libgpmlog.cuh"

#include "gmemtable.h"
#include "debugger.h"
#include "gpu_batch.cuh"
#include "gpu_updates.cuh"
#include "helper.cuh"
#include "log.cuh"


#define NTHREADS_PER_BLK 1024
#define NBLKS 72

#define TIME_NOW std::chrono::high_resolution_clock::now()

GpuUpdates::GpuUpdates(GpuUpdateBatch* batch, GMemtable** activeTable, GMemtable** immutableTables, GMemtableLog* gLog) : batch(batch), activeTable(activeTable), gLog(gLog), immutableTables(immutableTables) {
    keyLength = batch->keyLength;
    numUpdates = (*activeTable)->size;
    Debugger debug(DEBUG); 
    allocateMemory((void**)&gIndices, numUpdates * sizeof(uint64_t));
}


GpuUpdates::~GpuUpdates() {
    keyLength = 0;
    numUpdates = 0;
}

__device__ __forceinline__
int gStrcmp(char* str1, char* str2, int length) {
    for (int i = 0; i < length; i++) {
        if (str1[i] != str2[i]) {
            return (str1[i] < str2[i]) ? -1 : 1;
        }
    }
    return 0;
}

__device__
int updateHandleCmpEqual(char* key, GMemtable** table, char* value, uint64_t opID, uint64_t mid, int batchID) {
    value = (*table)->valuePointer[mid]; // Set initial value
    uint64_t bestMid = mid; // Variable to track the index of the best key
    // Check if the current batchID matches the table's batchID
    // Return the index of the key with the largest opID less than the given opID
    // within the same batchID and where keys are equal
    if ((*table)->batchID == batchID) {
        while (mid + 1 < (*table)->size) { // Ensure we do not go out of bounds
            char* nextKey = (*table)->keys + (mid + 1) * (*table)->keyLength;
            if (gStrcmp(key, nextKey, (*table)->keyLength) == 0) { // Continue while the next key is equal to the current key
                if ((*table)->opID[mid + 1] < opID) {
                    bestMid = mid + 1; // Update bestMid as we found a closer match
                    value = (*table)->valuePointer[bestMid]; // Update the value pointer to the best match
                }
                mid++; // Move to the next mid
            } else {
                break; // Break the loop if the next key does not match
            }
        }
    } else {
        while(mid + 1 < (*table)->size) {
            char* nextKey = (*table)->keys + (mid + 1) * (*table)->keyLength;
            if (gStrcmp(key, nextKey, (*table)->keyLength) == 0) {
                bestMid = mid + 1; 
                mid++;
            } else {
                break;
            }            
        }
    }
    return bestMid;
}

__device__ void intToString(int input, char *output, int numDigits) {
    int i = numDigits - 1;
    while (input > 0 && i >= 0) {
        output[i--] = '0' + (input % 10); // Add digit to output string
        input /= 10;
    }
    while (i >= 0) {
        output[i--] = '0'; // Add leading zeros to output string
    }
    output[numDigits] = '\0'; // Add null terminator to output string
#ifdef __PRINT_DEBUG__
    printf("%s\t", output); 
#endif
}

__device__ int stringToInt(char *input, int valueLength) {
    int output = 0; 
    if(input == NULL)
        return 0; 
#ifdef __PRINT_DEBUG__
    printf("TID: %d input: %s valueLen: %d\n", threadIdx.x, input, valueLength); 
#endif
    for(int i = 0; i < valueLength - 1; ++i) {
        if (*(input + i) >= '0' && *(input + i) <= '9') {
            output = (output * 10) + (input[i] - '0'); // Convert digit to integer
        } else {
            break; // Stop processing if non-digit character encountered
        }
    }
    return output; 
}



__device__
uint64_t updateLookupKey(char* key, GMemtable** table, char* value, uint64_t opID, int batchID) {
    uint64_t start = 0;
    uint64_t end = (*table)->size - 1;
    uint64_t mid = 0;
    while (start <= end) {
        mid = start + (end - start) / 2;
        char* midKey = (*table)->keys + mid * (*table)->keyLength;
        int cmp = gStrcmp(key, midKey, (*table)->keyLength);
        if (cmp == 0) {
            value = (*table)->valuePointer[mid];
            
            // printf("OpID: %d, key: %s, midKey: %s\n", opID, key, midKey);
            // return mid; 
            return updateHandleCmpEqual(key, table, value, opID, mid, batchID); // TODO: fix this
        } else if (cmp < 0) {
            end = mid - 1;
        } else {
            start = mid + 1;
        }
    }
    return 0;
}

// __device__  
// unsigned int updateHash(const char* key, unsigned int numSets) {
//     unsigned long int value = 0;
//     for (int i = 0; key[i] != '\0'; i++) {
//         value = value * 37 + key[i];
//     }
//     return value % numSets;
// }

__device__ 
void incrementKey(GMemtable* table, char* keys, char* valuePtr, uint64_t opID, int batchID, char* valueToBeUpdates) {
    int valueLength = 8; 
    uint64_t value = stringToInt(valuePtr, valueLength); 
    char newValue[8]; 
    value++;
    intToString(value, newValue, valueLength);
    memcpy(valueToBeUpdates, newValue, valueLength);

}

__global__
void updateIndicesKernel(GpuUpdateBatch* batch, GMemtable* table, GMemtable* immutableTable, GMemtableLog* gLog, uint64_t* dSortedIndices, uint64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= size) return;

    int keyLength = batch->keyLength;

    for(uint64_t i = idx; i < size; i += NTHREADS_PER_BLK * NBLKS) {
        int sortedIndex = dSortedIndices[i]; 
        
        memcpy(table->keys + i * keyLength, batch->keys + sortedIndex * keyLength, keyLength);
        cudaMemcpyAsync(gLog->keys + i * keyLength, batch->keys + sortedIndex * keyLength, keyLength, cudaMemcpyDeviceToHost);
        
        table->opID[i] = batch->opID[sortedIndex];
        gLog->opID[i]= batch->opID[sortedIndex];  

        int valueLength = 8;

        updateLookupKey(batch->keys + sortedIndex * keyLength, &immutableTable, batch->valuesPtr + sortedIndex, batch->opID[sortedIndex], batch->batchID);  
        incrementKey(table, batch->keys + sortedIndex * keyLength, batch->valuesPtr + sortedIndex, batch->opID[sortedIndex], batch->batchID, &table->values[i * valueLength]);
        // This needs to be modified
        table->valuePointer[i] = &batch->valuesPtr[sortedIndex];
        gLog->valuePtrs[i] = &batch->valuesPtr[sortedIndex];
        
        // printf("Key: %s\n", table->keys + i * keyLength);
        gpm_drain(); 
    }
}


//void sortPutsOnGPU(char *putKeys, char **putValuePointers, uint64_t *putOperationIDs, int numUpdates, int keyLength, Memtable& table, int batchID) 
void GpuUpdates::updatesOnGpu() {
    GMemtable* table = *activeTable; 
    allocateMemory((void**)&gIndices, numUpdates * sizeof(uint64_t));
    cudaMemset(gIndices, 0, numUpdates * sizeof(uint64_t));
    // char* keys = batch->keys; 
    
    debug.setDebugMode(DEBUG);
    // Sort the keys in the batch and insert them into the memtable
    // The sort functions give the indices for the sorted key list and not the sorted keys themselves

    if (numUpdates < SORT_LIMIT) {
        sortSmallerUpdates(gIndices);
    } else {
        sortLargerUpdates(gIndices); 
    }

    debug.print ("******************Sorted keys on GPU");
    // Rearrange the value pointers and operation IDs based on the sorted index array
    cudaError_t err = cudaPeekAtLastError();
    debug.print("Error: " + to_string(err)); 

    // The sorted indices are used to setup the memtable and WAL

    updateIndicesKernel<<<NBLKS, NTHREADS_PER_BLK>>>(batch, table, immutableTables[0], gLog, gIndices, numUpdates); 
    cudaDeviceSynchronize();
    debug.print("********************Updated memtable and WAL with sorted keys");
    err = cudaPeekAtLastError();
    debug.print("Error: " + to_string(err)); 

    // Copy table->keys to CPU memory and print them
#if DEBUG
    char* cpuKeys = new char[keyLength * numUpdates];
    cudaMemcpy(cpuKeys, table->keys, keyLength * numUpdates, cudaMemcpyDeviceToHost);
    for (int i = 0; i < numUpdates; i++) {
        std::cout << "[DEBUG]: " << "Key " << i << ": " << cpuKeys + i * keyLength << std::endl;
    }
    delete[] cpuKeys; 
#endif

}

