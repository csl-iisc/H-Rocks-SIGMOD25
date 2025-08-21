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
#include <algorithm>

#define EXPECTED_KEY_LEN 8
#define BUFFER_FRACTION 0.5

UpdateSubBatch::UpdateSubBatch(int batchID, uint64_t updateBatchSize, Config config, GMemtable** activeTable, 
        GMemtable** immutableTables, BlockCache* cache, WriteSubBatch& writeBatch, rocksdb::DB* db, 
        std::string fileLocation, DbTimer* timer) : batchID(batchID), numUpdates(0), updateBatchSize(updateBatchSize), config(config), 
        activeTable(activeTable), immutableTables(immutableTables), cache(cache), db(db), fileLocation(fileLocation), timer(timer) {
    keys.reserve(updateBatchSize * EXPECTED_KEY_LEN); 
    Debugger debug(DEBUG);
    debug.print("Update sub batch created");
    numUpdates = 0; 

    // The code is implemented assuming we have only UPDATE and PUT ops s
}

//     // Destructor
UpdateSubBatch::~UpdateSubBatch() {
    keys.clear();
    opIDArr.clear();
}

    // Methods to add operations can be added here
void UpdateSubBatch::addUpdateOperation(std::string key, uint64_t opID) {
    keys.insert(keys.end(), key.begin(), key.end());
    keyLength = strlen(key.c_str()); 
    numUpdates++;
    opIDArr.push_back(opID); 
    debug.print("Update operation added to the update sub batch");
    debug.print("Updating: " + key);
}

uint64_t UpdateSubBatch::getNumUpdates() {
    return numUpdates;
}


void UpdateSubBatch::allocateAndSetupTable(GMemtable* table, uint64_t tableSize, uint64_t batchID, uint64_t keyLength, uint64_t memtableID) {
    
    (table)->size = tableSize; 
    (table)->numKeys = tableSize; 
    (table)->batchID = batchID; 
    (table)->keyLength = keyLength;
    (table)->memtableID  = memtableID; 

    allocateMemory((void**)&((table)->keys), tableSize * keyLength);  
    // cudaMalloc((void**)&((table)->keys), tableSize * keyLength);
    allocateMemory((void**)&((table)->valuePointer), tableSize * 8); 
    allocateMemory((void**)&((table)->opID), tableSize * sizeof(uint64_t));
}

// Push the memtable to the queue and notify the consumer thread
void UpdateSubBatch::pushMemtableToQueue(GMemtable* table) {
    std::unique_lock<std::mutex> lock(queueMutex);
    memtableQueue.push(table);
    queueCondVar.notify_one();
    lock.unlock();
}

// Convert the memtable to SST and write it to NVM
// Call the SST writer class to write the memtable to NVM
void UpdateSubBatch::convertMemtableToSST() {
    std::unique_lock<std::mutex> lock(queueMutex);
    // Wait until there is an item in the queue
    queueCondVar.wait(lock, [this]{ return !memtableQueue.empty(); });
    // At this point, we have the lock and there is at least one item in the queue.
    GMemtable* table = memtableQueue.front();
    memtableQueue.pop();
    lock.unlock(); // Unlock as soon as the critical section is over.
    // Assuming SstWriter does not need the lock to be held.
    SstWriter sstWriter(table, db, fileLocation);
}


void UpdateSubBatch::gMemtableAllocation() {
    // TODO: scale it for more update g-memtables 
    // For now, assuming that each update sub-batch generates one active table 
    // First push the active table to immutable table 
    immutableTables[0] = *activeTable;
    // Allocate memory for the active table
    timer->startTimer("memtableAlloc", memtableID);
    GMemtable* table;
    allocateMemoryManaged((void**)&table, sizeof(GMemtable)); 
    allocateAndSetupTable(*activeTable, updateBatchSize, batchID, keyLength, 0);
    *activeTable = table;
    timer->stopTimer("memtableAlloc", memtableID);

    GMemtableLog** gWAL;
    timer->startTimer("memtableAlloc", memtableID);
    allocateMemoryManaged((void**)&gWAL, sizeof(GMemtableLog));
    timer->stopTimer("memtableAlloc", memtableID);
    gWAL[0]->setupLog(fileLocation, *activeTable);

    

    // Push the active table to the queue
}

void UpdateSubBatch::gBatchAllocation() {

    timer->startTimer("gBatchAlloc", batchID);
    gBatch->cKeys = keys.data();
    cudaHostRegister(gBatch->cKeys, numUpdates * keyLength, cudaHostRegisterDefault);

    uint64_t* _opID = opIDArr.data();
    cudaHostRegister(_opID, numUpdates * sizeof(uint64_t), cudaHostRegisterDefault);

    debug.print("Size of keys: " + std::to_string(keys.size()));

    allocateMemory((void**)& gBatch->cKeys, numUpdates * keyLength);
    allocateMemory((void**)& gBatch->valuesPtr, numUpdates * sizeof(char*));
    allocateMemory((void**)& gBatch->opID, numUpdates * sizeof(uint64_t));
    debug.print("Allocated memory for keys, valuesPtr and opID in GPU Write Batch");

    copyMemoryAsync(gBatch->keys, gBatch->cKeys, numUpdates * keyLength, cudaMemcpyHostToDevice); 
    copyMemoryAsync(gBatch->opID, _opID, numUpdates * sizeof(uint64_t), cudaMemcpyHostToDevice); 
    
    gBatch->numUpdates = numUpdates;
    gBatch->keyLength = keyLength; 
    // gBatch->valueLength = valueLength; 
    gBatch->batchID = batchID;

    timer->stopTimer("gBatchAlloc", batchID);
}


void UpdateSubBatch::execute() {
    std::cout << "************* EXECUTING THE UPDATE SUB-BATCH ON THE GPU !****\n";
    // Determine the number of memtables needed for supporting the batch size 
    memtableConsumerThread = std::thread(&UpdateSubBatch::convertMemtableToSST, this); // Launch thread with convertMemtableToSST
    // memtableConsumerThread.detach(); 
    debug.setDebugMode(DEBUG);

    // TODO: change the fileLocation to what was provided by the user 
    // TODO: run this in parallel with another process

    debug.print("Executing the write sub batch");

    allocateMemoryManaged((void**)&gBatch, sizeof(GpuUpdateBatch));  
    
    gBatchAllocation();  
    if(persistThread.joinable()) {
        std::cout << "Persist thread\n"; 
        persistThread.join(); // Wait for persist thread to finish persisting here
    }
    memtableProducerThread = std::thread(&UpdateSubBatch::gMemtableAllocation, this); // join in destructor
    
    if (memtableProducerThread.joinable()) {
        std::cout << "memt producer thread\n"; 
        memtableProducerThread.join(); 
    }
    if (memtableConsumerThread.joinable()) {
        std::cout << "memt consumer thread\n"; 
        memtableConsumerThread.join(); 
    }
    
}
