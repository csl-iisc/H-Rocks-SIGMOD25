#include "sub_batch.h"
#include <iostream>
#include <vector>
#include "debugger.h"
#include "gmemtable.h"
#include "helper.cuh"
#include "log.cuh"
#include "gpu_batch.cuh"
#include "gpu_puts.cuh"
#include "gpu_gets.cuh"
#include "sst_writer.h"
#include "cpu_gets.h"
#include <thread>
#include "config.h"
#include "db_timer.h"

#define EXPECTED_KEY_LEN 16
#define BUFFER_FRACTION 0.5


ReadSubBatch::ReadSubBatch(int batchID, uint64_t readBatchSize, Config config, GMemtable** activeTable, GMemtable** immutableTables, 
        BlockCache* cache, WriteSubBatch& writeBatch, rocksdb::DB* db, std::string fileLocation, DbTimer* timer) : 
        batchID(batchID), numReads(0), readBatchSize(readBatchSize), config(config), activeTable(activeTable), 
        immutableTables(immutableTables), cache(cache), db(db), fileLocation(fileLocation), timer(timer) {
    // keys.reserve(readBatchSize * EXPECTED_KEY_LEN);
    this->writeBatch = &writeBatch;
    sharedBuffer = NULL; 
    debug.setDebugMode(DEBUG);
    numReads = 0; 
}

    // Destructor
ReadSubBatch::~ReadSubBatch() {
    keys.clear(); 
    opIDArr.clear();
    if (sharedBuffer != NULL) {
    freeMemory(sharedBuffer->notFoundKeysBitMap);
    freeMemory(sharedBuffer);
    }
}

    // Methods to add operations can be added here
void ReadSubBatch::addReadOperation(std::string key, uint64_t opID) {
    keys.insert(keys.end(), key.begin(), key.end());
    keys.push_back('\0');
    keyLength = strlen(key.c_str()) + 1; 
    opIDArr.push_back(opID); 
    numReads++;
    valueLength = 8; // TODO: set this properly later
}

uint64_t ReadSubBatch::getNumReads() {
    return numReads;
}

void ReadSubBatch::signalGpuProcessingComplete() {
    std::lock_guard<std::mutex> lock(mtx);
    gpuProcessingDone = true;
    cv.notify_one();  // Notify one waiting thread
}

// This was an alternate design choice which we did not go through with. 
// Ignore 
/*void ReadSubBatch::allocateSharedBuffer2() {
allocateMemoryHost((void**) &sharedBuffer2->notFoundKeysBuffer, sizeof(char) * readBatchSize * BUFFER_FRACTION * keyLength);
    allocateMemoryHost((void**) &sharedBuffer2->numNotFoundKeys, sizeof(uint64_t));
    allocateMemoryHost((void**) &sharedBuffer2->head, sizeof(uint64_t));
    allocateMemoryHost((void**) &sharedBuffer2->tail, sizeof(uint64_t));
    sharedBuffer2->bufferSize = readBatchSize * BUFFER_FRACTION * keyLength;
    debug.print("Not found keys buffer allocated with size: " + std::to_string(readBatchSize * BUFFER_FRACTION * keyLength));
    *(sharedBuffer2->numNotFoundKeys) = 0;
    *(sharedBuffer2->head) = 0;
    *(sharedBuffer2->tail) = 0; 
    sharedBuffer2->keyLength = keyLength;
    sharedBuffer2->batchID = batchID;
}
*/

void ReadSubBatch::allocateSharedBuffer() {
    // allocate  notFoundKeysBitMap 
    allocateMemory((void**) &sharedBuffer->notFoundKeysBitMap, sizeof(bool) * readBatchSize);
    allocateMemory((void**)& sharedBuffer->numNotFoundKeys, sizeof(uint64_t));
    allocateMemoryHost((void**) &sharedBuffer->doorbell, sizeof(int));
    sharedBuffer->readBatchSize = readBatchSize; 
}

void ReadSubBatch::execute() {
    *activeTable = *(writeBatch->activeTable); 
    *immutableTables = *(writeBatch->immutableTables);
    debug.setDebugMode(DEBUG);
    debug.print("Executing the read sub batch");
    cache = new BlockCache(keyLength, valueLength);
    debug.print("Block cache created");
   
    // Allocate the shared buffer
    // This was an alternate design which was not used -- ignore
    /*
    timer->startTimer("readSubBatchAlloc", batchID);
    allocateMemoryManaged((void**) &sharedBuffer2, sizeof(SharedBuffer));
    timer->stopTimer("readSubBatchAlloc", batchID);
    allocateSharedBuffer2(); // 
    debug.print("Shared buffer allocated" + std::to_string(readBatchSize * BUFFER_FRACTION * keyLength));
    */
    
    timer->startTimer("readSubBatchAlloc", batchID);
    allocateMemoryManaged((void**) &sharedBuffer, sizeof(SharedBuffer));
    allocateSharedBuffer(); 
    GpuGets gpuGets(this, activeTable, immutableTables, cache, sharedBuffer, timer);
    CpuGets cpuGets(keys, keyLength, db, sharedBuffer, cache, gpuProcessingDone, cv);
                                                                                                                                                                                                              
    timer->startTimer("gpuGets", batchID);
    gpuProcessingDone = false;

    timer->startTimer("cpuGets", batchID);
    // CpuGets cpuGets(db, sharedBuffer, valuePtrArr, cache, mtx, gpuProcessingDone, cv);
    // gGets.search();
    keyNotFoundProducerThread = std::thread(&GpuGets::search, &gpuGets); // join in destructor
    keyNotFoundConsumerThread = std::thread(&CpuGets::performGets, &cpuGets); // join in destructor

    // cpuGets.performGets();
    if (keyNotFoundProducerThread.joinable()) {
        keyNotFoundProducerThread.join();
    }
    timer->stopTimer("gpuGets", batchID);

    signalGpuProcessingComplete();
    if (keyNotFoundConsumerThread.joinable()) {
        keyNotFoundConsumerThread.join();
    }
    timer->stopTimer("cpuGets", batchID);

    std::cout << "readSubBatchAlloc time: " << timer->getTotalTime("readSubBatchAlloc") << std::endl;
    std::cout << "gpuGets time: " << timer->getTotalTime("gpuGets") << std::endl;
    std::cout << "cpuGets time: " << timer->getTotalTime("cpuGets") << std::endl;
}

UpdateSubBatch::UpdateSubBatch(int batchID, uint64_t updateBatchSize, Config config, GMemtable** activeTable, 
        GMemtable** immutableTables) : batchID(batchID), numUpdates(0), updateBatchSize(updateBatchSize), config(config), 
        activeTable(activeTable), immutableTables(immutableTables) {
    keys.reserve(updateBatchSize * EXPECTED_KEY_LEN); 
}



RangeSubBatch::RangeSubBatch(int batchID, uint64_t rangeBatchSize, Config config) : batchID(batchID), numRangeQueries(0), rangeBatchSize(rangeBatchSize), config(config) {
    startKeys.reserve(rangeBatchSize);
    endKeys.reserve(rangeBatchSize);
}

RangeSubBatch::~RangeSubBatch() {
    startKeys.clear();
    endKeys.clear();
    opIDArr.clear();
}

void RangeSubBatch::addRangeOperation(std::string startKey, std::string endKey, uint64_t opID) {
    startKeys.insert(startKeys.end(), startKey.begin(), startKey.end());
    endKeys.insert(endKeys.end(), endKey.begin(), endKey.end());
    opIDArr.push_back(opID); 
    keyLength = strlen(startKey.c_str());   
    numRangeQueries++;
}

uint64_t RangeSubBatch::getNumRangeQueries() {
    return numRangeQueries;
}