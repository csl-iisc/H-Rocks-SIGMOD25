
#include <cstdlib>
#include <math.h>

#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <getopt.h>
#include <unistd.h>

#include "gmemtable.h"
#include "gpu_batch.cuh"
#include "debugger.h"
#include "helper.cuh"
#include "gpu_puts.cuh"

#define __PRINT_DEBUG__ true

#define NTHREADS 128
#define MAX_THREADS_PER_BLOCK 1024

// #define __PRINT_DEBUG__ true

double calculateDiff (struct timespec t1, struct timespec t2) { 
    return (((t1.tv_sec - t2.tv_sec)*1000.0) + (((t1.tv_nsec - t2.tv_nsec)*1.0)/1000000.0));
}

__global__ void findSuccessor( unsigned char *dArrayStringVals, unsigned long long int *dArraySegmentKeys,  
        unsigned int *dArrayValIndex, unsigned long long int *dArraySegmentKeysOut,  unsigned int numKeys, 
        unsigned int stringSize, unsigned int charPosition, unsigned int segmentBytes) {

    int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
    if(threadID > numKeys) return;
    dArraySegmentKeysOut[threadID] = 0;

    if(threadID > 0) { 
        if(dArraySegmentKeys[threadID] != dArraySegmentKeys[threadID-1]) { 
            dArraySegmentKeysOut[threadID] = ((unsigned long long int)(1) << 56);
        }
    }

    unsigned int stringIndex = dArrayValIndex[threadID];
    unsigned long long int currentKey = (dArraySegmentKeys[threadID] << (segmentBytes*8));
    unsigned char ch;
    int i = 0;
    unsigned int end = 0;

    for(i = 7; i >= ((int)segmentBytes); i--) { 
        ch = (unsigned char)(currentKey >> (i*8));
        if(ch == '\0') { 
            end = 1;
            break;
        }
    }

    if( end == 0) {
        unsigned int startPosition = charPosition;
        for(i = 6; i >=0; i--) { 
            if( stringIndex +  startPosition < stringSize ) { 
                ch = dArrayStringVals[ stringIndex + startPosition ];
                dArraySegmentKeysOut[threadID] |= ((unsigned long long int) ch << (i*8)); 
                startPosition++;
                if(ch == '\0') break;
            }
            if(ch == '\0') break;
        }

    } else { 
        dArraySegmentKeysOut[threadID] = ((unsigned long long int)(1) << 56);
    }
}

__global__ void  eliminateSingleton(unsigned int *dArrayOutputvalIndex, unsigned int *dArrayValIndex, unsigned int *dArrayStaticIndex, 
        unsigned int *d_array_map, unsigned int *dArrayStencil, int currentSize) {

    int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
    if(threadID >= currentSize) return;

    dArrayStencil[threadID] = 1;

    if(threadID == 0 && (d_array_map[threadID + 1] == 1)) { 
        dArrayStencil[threadID] = 0; 
    } else if( (threadID == (currentSize-1)) && (d_array_map[threadID] == 1) ) {
        dArrayStencil[threadID] = 0;  
    } else if( (d_array_map[threadID] == 1) && (d_array_map[threadID + 1] == 1)) { 
        dArrayStencil[threadID] = 0; 
    }

    if(dArrayStencil[threadID] == 0) { 
        dArrayOutputvalIndex[ dArrayStaticIndex[threadID] ] = dArrayValIndex[threadID]; 
    }
}

__global__ void rearrangeSegMCU(unsigned long long int *dArraySegmentKeys, unsigned long long int *dArraySegmentKeysOut, 
        unsigned int *dArraySegment, unsigned int segmentBytes, unsigned int numKeys) { 

    int threadID = (blockIdx.x * blockDim.x) +  threadIdx.x;
    if(threadID >= numKeys) return;

    unsigned long long int currentKey = (dArraySegmentKeysOut[threadID] << 8);
    unsigned long long int segmentID  = (unsigned long long int) dArraySegment[threadID];
    dArraySegmentKeys[threadID] = (segmentID << ((8-segmentBytes)*8));
    dArraySegmentKeys[threadID] |= (currentKey >> (segmentBytes*8));
    return;
}

struct getSegmentBytes {
    __host__ __device__
        unsigned int operator()(const unsigned long long int& x) const { 
            return (unsigned int)(x >> 56);
        }
};

void printChars(unsigned long long int val, unsigned int segmentBytes) { 
    Debugger debug(DEBUG);  

    debug.print("printing keys");
    int shift = 56;
    if(segmentBytes > 0) { 
        debug.print("segment number " + std::to_string((unsigned int)(val>>((8-segmentBytes)*8))));
        shift-=(segmentBytes*8);
    }
    while(shift>=0) {
        char ch = (char)(val>> shift);
        debug.print(std::string(1, ch));
        shift-=8;
        if(ch == '\0') {
            debug.print("*");
        }
    }
    debug.print(" ");
}


void initializeBuckets(unsigned char* inbuf, int keyLen, int numKeys, thrust::host_vector<unsigned char> hStringVals, thrust::host_vector<unsigned int> hValIndex, thrust::host_vector<unsigned long long int> hKeys) {
    //hStringVals.push_back(inbuf); 
    for(uint64_t i = 0; i < numKeys; ++i) {
        // debug.print("Copying key: " + std::to_string(i)); 
        hValIndex[i] = i * keyLen; 
        hStringVals[(i+1) * keyLen] = '\0'; 
        unsigned int prefixLen = 0;
        unsigned long long int firstKey = 0;
        for (int j = 0; j < min(keyLen , 8); j++) {
            firstKey |= (((unsigned long long int) hStringVals[i * keyLen + j] << (7 - prefixLen) * 8)); 
            prefixLen++; 
        }
        hKeys[i] = firstKey; 
    }
}

// TODO: when calling this function, pass the starting point in the batch for sorting instead of the 0th element of the batch everytime 
// TODO: this function get iteratively called till all the sorting is done 

void GpuPuts::sortLarger(uint64_t* gIndices) {
    thrust::device_vector<uint64_t> indices(numWrites);
    Debugger debug(DEBUG);  

    uint64_t numKeys = (*activeTable)->size; 
    uint32_t keyLength = (*activeTable)->keyLength;
    char* keys = batch->cKeys;
    debug.print("Number of keys: " + std::to_string(numKeys));

    thrust::host_vector<unsigned long long int> hKeys(numKeys);
    thrust::host_vector<unsigned int> hValIndex(numKeys);
    thrust::host_vector<unsigned char> hStringVals(numKeys * keyLength + 2); 
    debug.print("Keys copied to host vector");

    // Can later pass the file name 
    thrust::copy(keys, keys + numKeys * keyLength, hStringVals.begin());

#pragma omp parallel for num_threads(NTHREADS)
    for(uint64_t i = 0; i < numKeys; ++i) {
        hValIndex[i] = i * keyLength; 
        unsigned long long int firstKey = 0;
        for (unsigned int prefixLen = 0; prefixLen < min(keyLength, 8); prefixLen++) {
            unsigned char ch = (unsigned char) keys[i * keyLength + prefixLen]; 
            firstKey |= (((unsigned long long int) ch) << ((7 - prefixLen) * 8)); 
        }
        hKeys[i] = firstKey;
    }

    thrust::device_vector<unsigned char> dStringVals = hStringVals;
    thrust::device_vector<unsigned long long int> dSegmentKeys = hKeys;
    thrust::device_vector<unsigned int> dValIndex = hValIndex;
    thrust::device_vector<unsigned int> dStaticIndex(numKeys);
    thrust::device_vector<unsigned int> dOutputValIndex(numKeys);

    thrust::sequence(dStaticIndex.begin(), dStaticIndex.begin() + numKeys);

    cudaError_t err = cudaPeekAtLastError();
    debug.print("Error: " + std::to_string(err));

    unsigned int charPosition = 8;
    unsigned int originalSize = numKeys;
    unsigned int segmentBytes = 0;
    unsigned int lastSegmentID = 0;

    unsigned int numSorts = 0;
    unsigned char* dArrayStringVals; 
    unsigned int* dArrayValIndex; 


    while(true) { 

        thrust::sort_by_key (
                dSegmentKeys.begin(),
                dSegmentKeys.begin() + numKeys,
                dValIndex.begin()
                ); 
        numSorts++;

        thrust::device_vector<unsigned long long int> dSegmentKeysOut(numKeys, 0);

        dArrayStringVals = thrust::raw_pointer_cast(&dStringVals[0]); 
        dArrayValIndex = thrust::raw_pointer_cast(&dValIndex[0]);
        unsigned int *dArrayStaticIndex = thrust::raw_pointer_cast(&dStaticIndex[0]);
        unsigned int *dArrayOutputvalIndex = thrust::raw_pointer_cast(&dOutputValIndex[0]);

        unsigned long long int *dArraySegmentKeysOut = thrust::raw_pointer_cast(&dSegmentKeysOut[0]);
        unsigned long long int *dArraySegmentKeys = thrust::raw_pointer_cast(&dSegmentKeys[0]); 

        int numBlocks = 1;
        int numThreadsPerBlock = numKeys/numBlocks;

        if(numThreadsPerBlock > MAX_THREADS_PER_BLOCK) { 
            numBlocks = (int)ceil(numThreadsPerBlock/(float)MAX_THREADS_PER_BLOCK);
            numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
        }
        dim3 grid(numBlocks, 1, 1);
        dim3 threads(numThreadsPerBlock, 1, 1); 

        synchronizeDevice();
        debug.print("Grid: " + std::to_string(numBlocks) + " threads: " + std::to_string(numThreadsPerBlock));
        debug.print("************ Calling findSuccessor kernel");
        findSuccessor<<<grid, threads, 0>>>(dArrayStringVals, dArraySegmentKeys, dArrayValIndex, 
            dArraySegmentKeysOut, numKeys, keyLength, charPosition, segmentBytes);
        synchronizeDevice();
        cudaError_t err = cudaPeekAtLastError();
        debug.print("Error: " + std::to_string(err)); 
        
        charPosition+=7;

        thrust::device_vector<unsigned int> dTempVector(numKeys);
        thrust::device_vector<unsigned int> dSegment(numKeys);
        thrust::device_vector<unsigned int> dStencil(numKeys);
        thrust::device_vector<unsigned int> dMap(numKeys);

        unsigned int *dArrayTempVector = thrust::raw_pointer_cast(&dTempVector[0]);
        unsigned int *dArraySegment = thrust::raw_pointer_cast(&dSegment[0]);
        unsigned int *dArrayStencil = thrust::raw_pointer_cast(&dStencil[0]);


        thrust::transform(dSegmentKeysOut.begin(), dSegmentKeysOut.begin() + numKeys, dTempVector.begin(), getSegmentBytes());
        
#ifdef __PRINT_DEBUG__
        thrust::device_vector<unsigned int>::iterator itr;
        thrust::device_vector<unsigned long long int>::iterator itr2;
        thrust::device_vector<unsigned long long int>::iterator itr3;


        itr2 = dSegmentKeysOut.begin();
        itr3 = dSegmentKeys.begin();

        for(itr = dTempVector.begin(); itr!=dTempVector.end(); ++itr) { 
            std::cout << *itr << " ";
            printChars(*itr3, segmentBytes);
            std::cout << " ";
            printChars(*itr2, 1);
            ++itr2;
            ++itr3;
            std::cout << std::endl;
        }
#endif

        thrust::inclusive_scan(dTempVector.begin(), dTempVector.begin() + numKeys, dSegment.begin());

        synchronizeDevice(); 
        eliminateSingleton<<<grid, threads, 0>>>(dArrayOutputvalIndex, dArrayValIndex, dArrayStaticIndex, 
                dArrayTempVector, dArrayStencil, numKeys); 
        synchronizeDevice();

#ifdef __PRINT_DEUBG__
        std::cout << "Stencil values are ";
        for( itr = dStencil.begin(); itr != dStencil.end(); ++itr) { 
            std::cout << *itr << " ";
        }
        std::cout << endl;
#endif

        thrust::exclusive_scan(dStencil.begin(), dStencil.begin() + numKeys, dMap.begin());

        thrust::scatter_if(dSegment.begin(), dSegment.begin() + numKeys, dMap.begin(), 
                dStencil.begin(), dTempVector.begin());
        thrust::copy(dTempVector.begin(), dTempVector.begin() + numKeys, dSegment.begin()); 

        thrust::scatter_if(dValIndex.begin(), dValIndex.begin() + numKeys, dMap.begin(), 
                dStencil.begin(), dTempVector.begin());
        thrust::copy(dTempVector.begin(), dTempVector.begin() + numKeys, dValIndex.begin()); 

        thrust::scatter_if(dStaticIndex.begin(), dStaticIndex.begin() + numKeys, dMap.begin(), 
                dStencil.begin(), dTempVector.begin());
        thrust::copy(dTempVector.begin(), dTempVector.begin() + numKeys, dStaticIndex.begin()); 

        thrust::scatter_if(dSegmentKeysOut.begin(), dSegmentKeysOut.begin() + numKeys, dMap.begin(), 
                dStencil.begin(), dSegmentKeys.begin());
        thrust::copy(dSegmentKeys.begin(), dSegmentKeys.begin() + numKeys, dSegmentKeysOut.begin()); 

        numKeys = *(dMap.begin() + numKeys - 1) + *(dStencil.begin() + numKeys - 1); 
        if(numKeys != 0) { 
            lastSegmentID = *(dSegment.begin() + numKeys - 1);
        }

        dTempVector.clear();
        dTempVector.shrink_to_fit();

        dStencil.clear();
        dStencil.shrink_to_fit();

        dMap.clear();
        dMap.shrink_to_fit();

        if(numKeys == 0) {
            // This is not needed as we are copying the values to indices
            // thrust::copy(dOutputValIndex.begin(), dOutputValIndex.begin() + originalSize, hValIndex.begin());
            // Copy dOutputValIndex to indices 
            thrust::copy(dOutputValIndex.begin(), dOutputValIndex.begin() + originalSize, indices.begin());
            gIndices = thrust::raw_pointer_cast(indices.data());
            break;
        }

        segmentBytes = (int) ceil(((float)(log2((float)lastSegmentID+2))*1.0)/8.0);
        debug.print("segmentBytes: " + std::to_string(segmentBytes));
        charPosition-=(segmentBytes-1);

#ifdef __PRINT_DEBUG__
        printf("[DEBUG] numKeys %d, charPosition %d, lastSegmentID %d, segmentBytes %d\n", numKeys, 
                charPosition, lastSegmentID, segmentBytes );
#endif

        int numBlocks1 = 1;
        int numThreadsPerBlock1 = numKeys/numBlocks1;

        if(numThreadsPerBlock1 > MAX_THREADS_PER_BLOCK) { 
            numBlocks1 = (int)ceil(numThreadsPerBlock1/(float)MAX_THREADS_PER_BLOCK);
            numThreadsPerBlock1 = MAX_THREADS_PER_BLOCK;
        }
        dim3 grid1(numBlocks1, 1, 1);
        dim3 threads1(numThreadsPerBlock1, 1, 1); 

        synchronizeDevice();
        rearrangeSegMCU<<<grid1, threads1, 0>>>(dArraySegmentKeys, dArraySegmentKeysOut, dArraySegment, 
                segmentBytes, numKeys);
        synchronizeDevice();



#ifdef __PRINT_DEBUG__		
        printf("---------- new keys are --------\n");
        itr2 = dSegmentKeys.begin();
        unsigned int ct = 0;
        for( ct = 0; ct < numKeys; ct++ ) { 
            printChars(*itr2, segmentBytes);
            printf("\n");
            ++itr2;
        }
        printf("----\n");
#endif
    }
}