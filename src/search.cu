#include "gpu_gets.cuh"
#include <iostream>
#include "sub_batch.h"
#include <stdio.h>  
#include "gmemtable.h"
#include "block_cache.h"

#define NBLKS 72
#define NTHREADS_PER_BLK 1024

__device__ __forceinline__
int gStrcmp(char* str1, char* str2, int length) {
    for (int i = 0; i < length; i++) {
        if (str1[i] != str2[i]) {
            return (str1[i] < str2[i]) ? -1 : 1;
        }
    }
    return 0;
}


__device__ __forceinline__
int handleCmpEqual(char* key, GMemtable** table, char** value, uint64_t opID, uint64_t mid, int batchID) {
    *value = (*table)->valuePointer[mid]; // Set initial value
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
                    *value = (*table)->valuePointer[bestMid]; // Update the value pointer to the best match
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

__device__ __forceinline__
int lookupKey(char* key, GMemtable** table, char** value, uint64_t opID, int batchID) {
    uint64_t start = 0;
    uint64_t end = (*table)->size - 1;
    uint64_t mid = 0;
    while (start <= end) {
        mid = start + (end - start) / 2;
        char* midKey = (*table)->keys + mid * (*table)->keyLength;
        int cmp = gStrcmp(key, midKey, (*table)->keyLength);
        if (cmp == 0) {
            return handleCmpEqual(key, table, value, opID, mid, batchID);
        } else if (cmp < 0) {
            end = mid - 1;
        } else {
            start = mid + 1;
        }
    }
    return 0;
}

__device__  
    unsigned int hash(const char* key, unsigned int numSets) {
        unsigned long int value = 0;
        for (int i = 0; key[i] != '\0'; i++) {
            value = value * 37 + key[i];
        }
        return value % numSets;
    }



// lookup key in block cache if not found in memtable
__device__ 
char* lookupBlockCache(char* key, BlockCache* blockCache) {
    // Check if the key is in the block cache
    // If the key is in the block cache, then return the value
    // If the key is not in the block cache, then
    // return NULL
    uint64_t index = hash(key, blockCache->numSets);
    // Check if the key is in the block cache set 
    CacheSet* set = &blockCache->sets[index];
    for (int i = 0; i < blockCache->setSize; i++) {
        CacheEntry* entry = set->frequencyList[i];
        if (entry != NULL) {
            if (gStrcmp(key, entry->key, blockCache->keyLength) == 0) {
                return entry->value;
            }
        }
    }
    return NULL;
}

/*
__device__
void insertNotFoundKey (char* key, SharedBuffer* sharedBuffer) {
    uint64_t requiredHeadAdvance = sharedBuffer->keyLength;  // Amount of space needed
    uint64_t currentHead;
    uint64_t nextHead;
    bool spaceAvailable = false;

    do {
        currentHead = *sharedBuffer->head;
        nextHead = (currentHead + requiredHeadAdvance) % sharedBuffer->bufferSize;
        
        // Check if there is enough space to write the new key
        if (nextHead > currentHead) {
            // Normal case: head is behind tail
            spaceAvailable = nextHead < *sharedBuffer->tail || *sharedBuffer->tail <= currentHead;
        } else {
            // Wrap-around case: head will wrap and must not catch up to tail
            spaceAvailable = *sharedBuffer->tail != 0 && *sharedBuffer->tail <= currentHead;
        }

        // Busy-wait if there is no space; otherwise, exit loop
        if (!spaceAvailable) {
            printf("CPU is lagging behind. Increase the buffer size!!!\n"); 
            continue;
        }
    
    memcpy(sharedBuffer->notFoundKeysBuffer + currentHead, key, sharedBuffer->keyLength);
    atomicAdd(reinterpret_cast<unsigned int*>(sharedBuffer->numNotFoundKeys), 1); 
    __threadfence_system();  // Ensure global memory visibility

    } while (!atomicCAS(reinterpret_cast<unsigned long long*>(sharedBuffer->head), currentHead, nextHead));  // Atomically update head if still valid
    // Proceed with the insertion
}
*/

__device__
void insertKeyNotFound2(uint64_t location, SharedBuffer* sharedBuffer) {
    // Insert the key at the given location in the keyNotFoundBuffer
    // Increment the numKeyNotFound
    // printf("Inserting key at location: %ld\n", location);    
    sharedBuffer->notFoundKeysBitMap[location] = true;
    atomicAdd(reinterpret_cast<unsigned int*>(sharedBuffer->numNotFoundKeys), 1);
}

__device__ __forceinline__ void flush(SharedBuffer* sharedBuffer) {
    if (sharedBuffer->numNotFoundKeys == 0) {
        return;
    }
    // printf("Flushing %d bytes\n", sharedBuffer->numNotFoundKeys);
    while(atomicCAS(sharedBuffer->doorbell, 1, 0) != 1) {
        // Busy wait until the CPU resets the doorbell
    }
    *(sharedBuffer->doorbell) = 1; // Notify that there is something in the buffer. CPU can reset it to zero once it has consumed all the keys 
    /* make sure everything is visible in memory */
    __threadfence_system();

}

__global__
void binarySearch(GpuReadBatch* batch, GMemtable** activeTable, GMemtable** immutableTable, uint64_t numKeys, int keyLength, 
        int numTables, char** gValuePointersArr, BlockCache* blockCache, SharedBuffer* sharedBuffer) {
    uint64_t threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID >= numKeys) {
        return;
    }
    
    for(uint64_t idx = threadID; idx < numKeys; idx += blockDim.x * gridDim.x) {
        int batchID = batch->batchID;
        // Lookup the key in the activeTable and then all the immutableTables 
        // If the key is found in any of the tables, then return the value
        // If the key is not found in any of the tables, then return NULL
        char* key = batch->keys + idx * keyLength;
        char* value = NULL;
        uint64_t readOpID = batch->opID[idx];

        uint64_t index = lookupKey(key, activeTable, &value, readOpID, batchID); // sets the index and value

        if (value != NULL) {
            gValuePointersArr[idx] = value;
            return;
        }

        for (int i = 0; i < numTables - 1; i++) {
            index = lookupKey(key, &immutableTable[i], &value, readOpID, batchID);
            if (value != NULL) {
                gValuePointersArr[idx] = value;
                return;
            }
        }
        
        value = lookupBlockCache(key, blockCache);
        // copy key to keyNotFoundBuffer
        // atomic increment numKeyNotFound 
        if (value == NULL) {
            // printf("Current timestamp: %ld\n", time(NULL));
            // insertNotFoundKey(key, sharedBuffer);
            insertKeyNotFound2(idx, sharedBuffer); 
        }
        if (threadID == 0) { // Only one thread calls the flush function
            flush(sharedBuffer);
        }
    }
}



void GpuGets::search() {
    // Push the readSubBatch to GPU memory 
    allocateMemoryManaged((void**)&gReadBatch, sizeof(ReadSubBatch));
    gReadBatch->numReads = readBatch->numReads;
    gReadBatch->keyLength = readBatch->keyLength;
    gReadBatch->batchID = readBatch->batchID;
    
    char* keys = (char*)malloc(readBatch->numReads * readBatch->keyLength);
    uint64_t* opIDArr = (uint64_t*)malloc(readBatch->numReads * sizeof(uint64_t));
    char** outputValuesPtr = (char**)malloc(readBatch->numReads * sizeof(char*));

    cudaHostRegister(keys, readBatch->numReads * readBatch->keyLength, cudaHostRegisterDefault);
    cudaHostRegister(opIDArr, readBatch->numReads * sizeof(uint64_t), cudaHostRegisterDefault);
    // cudaHostRegister(outputValuesPtr, readBatch->numReads * sizeof(char*), cudaHostRegisterDefault);

    memcpy(keys, readBatch->keys.data(), readBatch->numReads * readBatch->keyLength);
    memcpy(opIDArr, readBatch->opIDArr.data(), readBatch->numReads * sizeof(uint64_t));
    // memcpy(outputValuesPtr, readBatch->outputValuesPtr, readBatch->numReads * sizeof(char*));

    allocateMemory((void**)&gReadBatch->keys, readBatch->numReads * readBatch->keyLength);
    allocateMemory((void**)&gReadBatch->opID, readBatch->numReads * sizeof(uint64_t));
    allocateMemory((void**)&gReadBatch->outputValuesPtr, readBatch->numReads * sizeof(char*));
    
    debug.print("Memory allocated for GPU readBatch\n"); 
    cudaError_t err = cudaGetLastError();
    debug.print("Error: " + std::to_string(err));

    copyMemory(gReadBatch->keys, keys, readBatch->numReads * readBatch->keyLength, cudaMemcpyHostToDevice);
    copyMemory(gReadBatch->opID, opIDArr, readBatch->numReads * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    debug.print("Copied readBatch to GPU memory\n");
    // err = cudaGetLastError();
    // debug.print("Error: " + std::to_string(err));

    int numMemtables = sizeof(immutableTable) / sizeof(immutableTable[0]);
    debug.print("Num immutable tables: " + std::to_string(numMemtables) + "\n");

    // Launch the kernel 
    timer->startTimer("search", readBatch->batchID);
    binarySearch<<<NBLKS, NTHREADS_PER_BLK>>>(gReadBatch, activeTable, immutableTable, numKeys, keyLength, numMemtables, 
        gReadBatch->outputValuesPtr, blockCache, sharedBuffer);
        
    cudaDeviceSynchronize();
    timer->stopTimer("search", readBatch->batchID);
    
    std::cout << "Search time: " << timer->getTotalTime("search") << "\n";

    debug.print("searching kernel done\n");
    err = cudaGetLastError();
    debug.print("Error: " + std::to_string(err));

#ifdef DEBUG
    // Print valuePointerArr after copying it to GPU memory
    char** hValuePointersArr = nullptr;
    allocateMemory((void**)&hValuePointersArr, readBatch->numReads * sizeof(char*));
    copyMemory(hValuePointersArr, gReadBatch->outputValuesPtr, readBatch->numReads * sizeof(char*), cudaMemcpyDeviceToHost);
    debug.print("Copied valuePointerArr from GPU memory\n");
    for (int i = 0; i < readBatch->numReads; i++) {
        char* valuePtr;
        debug.print("Value pointer at index " + std::to_string(i) + ": " + std::to_string(reinterpret_cast<uint64_t>(valuePtr)) + "\n");
    }
#endif
}