#include "cpu_gets.h"
#include <iostream>
#include "rocksdb/db.h"
#include <omp.h>
#include "rocksdb/slice.h"
#include <time.h>
#include <string>
#include <chrono> 
#include "db_timer.h"
#include "helper.cuh"

#define NTHREADS 64
#define TIME_NOW std::chrono::high_resolution_clock::now()

CpuGets::CpuGets(std::vector<char> allKeysToGet, int keyLength, rocksdb::DB* db, SharedBuffer* sharedBuffer, BlockCache* cache, 
        bool& gpuProcessingDone, std::condition_variable& cv) : 
        allKeysToGet(allKeysToGet), keyLength(keyLength), db(db), sharedBuffer(sharedBuffer), 
        blockCache(cache), gpuProcessingDone(&gpuProcessingDone), cv(&cv) {
    // Constructor implementation...
    Debugger debug(DEBUG);   
    timer = new DbTimer();
    _notFoundKeysBitMap = (bool*) malloc(sharedBuffer->readBatchSize * sizeof(bool));
}

#define NTHREADS 64
using namespace rocksdb;

void CpuGets::performGets() {
    // check if doorbell is 1 
    // if it is 1, then perform gets
    debug.setDebugMode(DEBUG);
    debug.print("CPU thread woke up NOW");

    // Wait until the GPU processing is done or there are keys in the buffer
    // wait for the doorbell to ring or the GPU processing to be done
    while(*(sharedBuffer->doorbell) != 1 && !(*gpuProcessingDone)); 
    // perform gets
    uint64_t *_numNotFoundKeys = (uint64_t*) malloc(sizeof(uint64_t));
    copyMemory(_notFoundKeysBitMap, sharedBuffer->notFoundKeysBitMap, sharedBuffer->readBatchSize * sizeof(bool), cudaMemcpyDeviceToHost); 
    copyMemory(_numNotFoundKeys, sharedBuffer->numNotFoundKeys, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    std::cout << "Number of keys not found: " << *_numNotFoundKeys << "\n";

    if (*_numNotFoundKeys == 0) {
        *(sharedBuffer->doorbell) = 0;
        return;
    }

    uint64_t keysLookedUp = 0;
#pragma omp parallel for num_threads(NTHREADS)
    for(uint64_t i = 0; i < sharedBuffer->readBatchSize; i++) {
        if(_notFoundKeysBitMap[i]) {
            std::string value;
            timer->startTimer("cpuGet", i);
            rocksdb::Status s = db->Get(rocksdb::ReadOptions(), rocksdb::Slice(allKeysToGet.data() + i * keyLength, keyLength), &value); 
            timer->stopTimer("cpuGet", i);
            if (s.ok()) {
                char* valuePtr = (char*)malloc(value.size());
                strncpy(valuePtr, value.c_str(), value.size());
                blockCache->insert(allKeysToGet.data() + i * keyLength, valuePtr); // Insert the key-value pair into the block cache
            }
            keysLookedUp++; 
        }
        // if (keysLookedUp == *_numNotFoundKeys) {
        //     break;
        // }
    }
    // lookup all the keys from the _notFOundKeysBitMap and store the values in the valuePtrs
    // The notFoundBitMap has index to the key notFound in the readSubBatch 
    *(sharedBuffer->doorbell) = 0;

}

CpuGets::~CpuGets() {
    free(_notFoundKeysBitMap);
}

CpuGets2::CpuGets2(rocksdb::DB* db, SharedBuffer2* sharedBuffer2, char** valuePtrArr, BlockCache* cache, 
        std::mutex& mtx, bool& gpuProcessingDone, std::condition_variable& cv)
        : db(db), sharedBuffer2(sharedBuffer2), valuePtrs(valuePtrArr), blockCache(cache), 
        mtx(&mtx), gpuProcessingDone(&gpuProcessingDone), cv(&cv) {
    // Constructor implementation...
    keyLength = sharedBuffer2->keyLength;
    Debugger debug(DEBUG);   
    timer = new DbTimer();
}


CpuGets2::~CpuGets2() {
    // Nothing to do here
}


// void CpuGets2::performGets() {
// #pragma omp parallel for num_threads(NTHREADS)
//     for (uint64_t i = 0; i < *numNotFoundKeys; i++) {
//         std::string value;
//         rocksdb::Status s = db->Get(rocksdb::ReadOptions(), rocksdb::Slice(notFoundKeys + i * keyLength, keyLength), &value); 
//         if (s.ok()) {
//             valuePtrs[i] = (char*)malloc(value.size());
//             strncpy(valuePtrs[i], value.c_str(), value.size());
//             blockCache->insert(notFoundKeys + i * keyLength, valuePtrs[i]); // Insert the key-value pair into the block cache
//         } else {
//             valuePtrs[i] = nullptr;
//         }
//     }
// }

void CpuGets2::performGets() {
    uint64_t keysRead = 0;
    std::unique_lock<std::mutex> lk(*mtx);

    while (!(*gpuProcessingDone)) {
        // Wait until the GPU processing is done or there are keys in the buffer
        (*cv).wait(lk, [this]{ return (*gpuProcessingDone) || *(sharedBuffer2->head) != *(sharedBuffer2->tail); });

        // debug.print("CPU thread woke up at: " + std::to_string(TIME_NOW));
        if (*(sharedBuffer2->head) != *(sharedBuffer2->tail)) {
            uint64_t start = *(sharedBuffer2->tail);
            uint64_t end = *(sharedBuffer2->head);
            for (uint64_t i = start; i < end; i++) {
                std::string value;
                timer->startTimer("cpuGet", keysRead);
                rocksdb::Status s = db->Get(rocksdb::ReadOptions(), rocksdb::Slice(sharedBuffer2->notFoundKeysBuffer + i * keyLength, keyLength), &value); 
                timer->stopTimer("cpuGet", keysRead);
                std::cout << "cpu get time: " << timer->getTotalTime("cpuGet") << "\n";
                if (s.ok()) {
                    valuePtrs[i] = (char*)malloc(value.size());
                    strncpy(valuePtrs[i], value.c_str(), value.size());
                    timer->startTimer("blockCacheInsert", keysRead);
                    blockCache->insert(sharedBuffer2->notFoundKeysBuffer + i * keyLength, valuePtrs[i]); // Insert the key-value pair into the block cache
                    timer->stopTimer("blockCacheInsert", keysRead);
                    std::cout << "block cache time: " << timer->getTotalTime("blockCacheInsert") << "\n";
                } else {
                    valuePtrs[i] = nullptr;
                }
            }
            keysRead += end - start;
            std::atomic<uint64_t>* tailPtr = reinterpret_cast<std::atomic<uint64_t>*>(sharedBuffer2->tail);
            tailPtr->store(end, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
        }

        // Optionally, recheck or reset conditions
        // Dry run and check if this is valid always 
        if ((*gpuProcessingDone)) break;  // Break out of the loop if GPU processing is confirmed done 
    }
    lk.unlock();
    debug.print("num keys read by the CPU are: " + std::to_string(keysRead));
}
