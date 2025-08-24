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

// WriteSubBatch::WriteSubBatch(int batchID, uint64_t writeBatchSize, Config config, GMemtable** activeTable, 
//         GMemtable** immutableTables, rocksdb::DB *db, std::string fileLocation, DbTimer* timer) : 
//         batchID(batchID), numWrites(0), writeBatchSize(writeBatchSize), config(config), activeTable(activeTable), 
//         immutableTables(immutableTables), db(db), fileLocation(fileLocation), timer(timer) {
//     keys.reserve(writeBatchSize * EXPECTED_KEY_LEN);
//     values.reserve(writeBatchSize * EXPECTED_KEY_LEN);
//     opIDArr.reserve(writeBatchSize);
//     Debugger debug(DEBUG); 
//     valuePtrArr = nullptr;
//     cudaFree(0);
//     memtableID = 0; 
//     debug.print("Write sub batch created"); 
// }

WriteSubBatch::WriteSubBatch(int batchID, uint64_t writeBatchSize, Config config,
    GMemtable** activeTable, GMemtable** immutableTables,
    rocksdb::DB* db, std::string fileLocation, DbTimer* timer,
    int& numMemtablesAcrossBatches,
    std::unordered_map<int,int>& memtableBatchMap)
  : batchID(batchID),
    numWrites(0),
    writeBatchSize(writeBatchSize),
    config(config),
    activeTable(activeTable),
    immutableTables(immutableTables),
    db(db),
    fileLocation(std::move(fileLocation)),
    timer(timer),
    numMemtablesAcrossBatches(numMemtablesAcrossBatches),
    memtableBatchMap(memtableBatchMap),
    valuePtrArr(nullptr),
    memtableID(0)
{
    // keys.reserve(writeBatchSize * EXPECTED_KEY_LEN);
    // values.reserve(writeBatchSize * EXPECTED_KEY_LEN);
    // opIDArr.reserve(writeBatchSize);
    Debugger debug(DEBUG);
    cudaFree(0);
    debug.print("Write sub batch created");
}

// WriteSubBatch::~WriteSubBatch() {
//     keys.clear();
//     values.clear();
//     opIDArr.clear();
//     keyLength = 0;
//     valueLength = 0;
//     numWrites = 0;
//     writeBatchSize = 0;
//     batchID = -1;
//     config = Config();
//     activeTable = nullptr;
//     immutableTables = nullptr;
//     // Debugger debug(DEBUG_MODE);
//     if (valuePtrArr != nullptr) {
//         cudaFree(valuePtrArr);
//     }
//     if (gBatch != nullptr) {
//         cudaFree(gBatch);
//     }
//     debug.print("Write sub batch destroyed");
// }

WriteSubBatch::~WriteSubBatch() {
    debug.setDebugMode(DEBUG);
    debug.print("WriteSubBatch dtor begin");

    // 1) Join all threads so no one still uses our memory
    if (persistThread.joinable())           persistThread.join();
    if (memtableProducerThread.joinable())  memtableProducerThread.join();
    if (memtableConsumerThread.joinable())  memtableConsumerThread.join();

    // 2) Ensure async copies done before unpin/free
    // cudaDeviceSynchronize();

    // 3) Unpin + free host-pinned staging buffers
    if (hostPinnedKeys)      { cudaHostUnregister(hostPinnedKeys);      free(hostPinnedKeys);      hostPinnedKeys = nullptr; }
    if (hostPinnedValuesPtr) { cudaHostUnregister(hostPinnedValuesPtr); free(hostPinnedValuesPtr); hostPinnedValuesPtr = nullptr; }
    if (hostPinnedOpID)      { cudaHostUnregister(hostPinnedOpID);      free(hostPinnedOpID);      hostPinnedOpID = nullptr; }

    // 4) Free CPU malloc’d value buffers
    if (valuePtrArr) { free(valuePtrArr); valuePtrArr = nullptr; }
    if (valueArr)    { free(valueArr);    valueArr    = nullptr; }

    // 5) Free device/UM allocations owned by gBatch, then gBatch itself
    // if (gBatch) {
    //     if (gBatch->keys)      { freeMemory(gBatch->keys);      gBatch->keys = nullptr; }
    //     if (gBatch->valuesPtr) { freeMemory(gBatch->valuesPtr); gBatch->valuesPtr = nullptr; }
    //     if (gBatch->opID)      { freeMemory(gBatch->opID);      gBatch->opID = nullptr; }
    //     freeMemory(gBatch);  // allocated via allocateMemoryManaged
    //     gBatch = nullptr;
    // }
    if (gBatch) { freeMemory(gBatch); gBatch = nullptr; }
    if (gLog)   { delete(gLog);   gLog   = nullptr; }
    
    // 6) Clear CPU vectors
    keys.clear();
    values.clear();
    opIDArr.clear();

    keyLength = valueLength = 0;
    numWrites = writeBatchSize = 0;
    batchID = -1;
    activeTable = nullptr;
    immutableTables = nullptr;

    debug.print("WriteSubBatch dtor end");
}

void WriteSubBatch::addWriteOperation(std::string key, std::string value, uint64_t opID) {
    keys.insert(keys.end(), key.begin(), key.end());
    keys.push_back('\0');
    values.insert(values.end(), value.begin(), value.end());
    values.push_back('\0');
    opIDArr.push_back(opID); 
    keyLength = strlen(key.c_str()) + 1; 
    valueLength = strlen(value.c_str()) + 1;
    numWrites++;
    // Debugger debug(DEBUG_MODE);
    debug.print("Write operation added to the write sub batch");
    debug.print("Inserting: " + std::string(key) + " : " + std::string(value) + " : " + std::to_string(numWrites) + " : " + std::to_string(writeBatchSize) + " : " + std::to_string(keyLength) + " : " + std::to_string(valueLength) + " : " + std::to_string(batchID) + " : " + std::to_string(opID));
}

uint64_t WriteSubBatch::getNumWrites() {
    return numWrites;
}

int WriteSubBatch::getValueLength() {
    return valueLength;
}

int WriteSubBatch::getKeyLength() {
    return keyLength;
}

char** WriteSubBatch::getValues() {
    return valuePtrArr;
}

void WriteSubBatch::valueArrConversion() {
    // allocate value ptr array 
    // convert the value vector into a value char array 
    // each pointer of the value pointer array points to the corresponding value in the value char array

    timer->startTimer("valueArrConversion", memtableID);
    valuePtrArr = (char**) malloc(sizeof(char*) * numWrites);
    valueArr = (char*) malloc(sizeof(char) * numWrites * valueLength);

    memcpy(valueArr, values.data(), numWrites * valueLength);
    // cudaHostRegister(valueArr, numWrites * valueLength * sizeof(char), cudaHostRegisterDefault); // TODO: uncomment this 

    debug.print("Value array conversion " + std::to_string(numWrites) + " : " + std::to_string(valueLength) + " : " + valueArr);
#pragma omp parallel for num_threads(100)
    for(int i = 0; i < (int)numWrites; i++) {
        valuePtrArr[i] = valueArr + i * valueLength; 
    }
    timer->stopTimer("valueArrConversion", memtableID);

    GMemtableLog* gLog = nullptr;
    // allocateMemoryManaged((void**)&gLog, sizeof(GMemtableLog)); 
    gLog = new GMemtableLog();
    // gLog->persistValues(fileLocation, batchID, numWrites, valueLength, valueArr); // TODO: do this in parallel by launching a separate thread
    persistThread = std::thread(&GMemtableLog::persistValues, gLog, fileLocation, batchID, numWrites, valueLength, valueArr);
    // persistThread.detach(); 
    debug.print("Persist thread launched and returned");
    // persistThread.join(); // TODO: move this join to its right place
    // debug.print("Persist thread joined");
}

void WriteSubBatch::allocateAndSetupTable(GMemtable* table, uint64_t tableSize, uint64_t batchID, uint64_t keyLength, uint64_t memtableID) {
    
    (table)->size = tableSize; 
    (table)->numKeys = tableSize; 
    (table)->batchID = batchID; 
    (table)->keyLength = keyLength;
    (table)->memtableID  = memtableID; 

    allocateMemory((void**)&((table)->keys), tableSize * keyLength);  
    allocateMemory((void**)&((table)->valuePointer), tableSize * 8); 
    allocateMemory((void**)&((table)->opID), tableSize * sizeof(uint64_t));
}

// Push the memtable to the queue and notify the consumer thread
void WriteSubBatch::pushMemtableToQueue(GMemtable* table) {
    std::unique_lock<std::mutex> lock(queueMutex);
    memtableQueue.push(table);
    queueCondVar.notify_one();
    lock.unlock();
}

// Convert the memtable to SST and write it to NVM
// Call the SST writer class to write the memtable to NVM
void WriteSubBatch::convertMemtableToSST() {
    std::unique_lock<std::mutex> lock(queueMutex);
    queueCondVar.wait(lock, [this]{ return !memtableQueue.empty(); });
    GMemtable* table = memtableQueue.front();
    memtableQueue.pop();
    lock.unlock();
    SstWriter sstWriter(table, db, fileLocation);  // frees table in its dtor
}




// Based on the numMemtablesAllocated and leftToAllocate we will decide how many memtables need to be evicted 
// When numMemtablesLeftToAllocate > config.maxMemtables, we will evict all the memtables and write them to disk
// However when numMemtablesLeftToAllocate < config.maxMemtables, we will evict only the required number of memtables
// And we will update the immutableTable pointers accordingly

void WriteSubBatch::memtableEviction(GMemtable* table, GMemtableLog* gWAL) {
    // Delete gWAL and call sstWriter for the table 
    
}

/*
// This function allocates and sets up memtables when the number of memtables needed are more than the ones specified in config 
void WriteSubBatch::allocateMultipleMemtablesMoreThanConfig(GpuWriteBatch* gBatch, int numMemtablesNeeded, 
        GMemtable**& immutableTables, int batchID, int keyLength, int& memtableID) {
    
    int numMemtablesLeftToAllocate = numMemtablesNeeded;
    int numMemtablesAllocated = 0;
    while (numMemtablesLeftToAllocate > 0) {
        debug.print("Number of memtables left to allocate: " + std::to_string(numMemtablesLeftToAllocate) + " Number of memtables allocated: " + std::to_string(numMemtablesAllocated));
        GMemtable* table;

        // allocate active table
        uint64_t activeTableSize = (numWrites < config.getMemtableSize()) ? numWrites : config.getMemtableSize();
        
        allocateMemoryManaged((void**)&table, sizeof(GMemtable)); 
        allocateAndSetupTable(table, activeTableSize, batchID, keyLength, memtableID);
        activeTable = &table;

        GMemtableLog** gWAL;
        allocateMemoryManaged((void**)&gWAL, sizeof(GMemtableLog) * numMemtablesNeeded);
        allocateMemoryManaged((void**)&gWAL[0], sizeof(GMemtableLog));
        gWAL[0]->setupLog(fileLocation, table);

        GpuPuts gPuts(gBatch, activeTable, gWAL[0]);
        gPuts.sortPutsOnGPU();

        // SstWriter sstWriter(*activeTable, db, fileLocation); // TODO: move out of the critical path and run in parallel
        // debug.print("****************SST writer object created");
        pushMemtableToQueue(*activeTable); 
            // Delete gWAL[0]
        memtableID++;
        numMemtablesAllocated++;
        numMemtablesLeftToAllocate--;

        // allocateMemoryManaged((void**)&immutableTables, sizeof(GMemtable*) * numMemtablesLeftToAllocate); // already allocated by batch constructor

        if (numMemtablesLeftToAllocate > config.maxMemtables) {
            debug.print("Number of memtables left to allocate is greater than the number of memtables allowed");
            allocateMemoryManaged((void**)&immutableTables, sizeof(GMemtable*) * config.maxMemtables);
            for(int i = 0; i < config.maxMemtables; i++) {
                uint64_t immutableMemtableSize = config.getMemtableSize(); 
                allocateMemoryManaged((void**)&table, sizeof(GMemtable)); 

                allocateAndSetupTable(table, immutableMemtableSize, batchID, keyLength, memtableID);
                immutableTables[i] = table;
                
                allocateMemoryManaged((void**)&gWAL[i + 1], sizeof(GMemtableLog));
                gWAL[i + 1]->setupLog(fileLocation, immutableTables[i]);
                GpuPuts gPuts(gBatch, &immutableTables[i], gWAL[i + 1]);
                gPuts.sortPutsOnGPU();
                // SstWriter sstWriter(*activeTable, db, fileLocation); // TODO: move out of the critical path and run in parallel
                // debug.print("****************SST writer object created");
                ToQueue(immutableTables[i]);
 
                memtableID++; 
                numMemtablesAllocated++;
                numMemtablesLeftToAllocate--;
                if (numMemtablesLeftToAllocate == 0) {
                    break;
                }
            }
        } else {
            debug.print("Number of memtables left to allocate is lesser than the number of memtables allowed");
            allocateMemoryManaged((void**)&immutableTables, sizeof(GMemtable*) * numMemtablesLeftToAllocate);
            for(int i = 0; i < numMemtablesLeftToAllocate; i++) {
                allocateMemoryManaged((void**)&table, sizeof(GMemtable));
                // For the last immutable memtable, the size could be less than the config size 
                uint64_t immutableMemtableSize = (i == numMemtablesLeftToAllocate - 1) ? 
                    (writeBatchSize - config.getMemtableSize() * numMemtablesAllocated) : config.getMemtableSize();
                allocateMemoryManaged((void**)&table, sizeof(GMemtable)); 

                allocateAndSetupTable(table, immutableMemtableSize, batchID, keyLength, memtableID);

                immutableTables[i] = table;

                // gWAL[i + 1]->setupLog(fileLocation, immutableTables[i]); // TODO: resolve file naming issue
                GpuPuts gPuts(gBatch, &immutableTables[i], gWAL[i + 1]);
                gPuts.sortPutsOnGPU();
                // SstWriter sstWriter(*activeTable, db, fileLocation); // TODO: move out of the critical path and run in parallel
                // debug.print("****************SST writer object created");
                pushMemtableToQueue(immutableTables[i]);
 
                memtableID++; 
                numMemtablesAllocated++;
            }
        }
    }
} 

void WriteSubBatch::allocateMultipleMemtablesLessThanConfig(GpuWriteBatch* gBatch, int numMemtablesNeeded, 
        GMemtable**& immutableTables, int batchID, int keyLength, int& memtableID) {

    uint64_t activeTableSize = (numWrites < config.getMemtableSize()) ? numWrites : config.getMemtableSize();
    GMemtable* table; 
    // Define gLog with so many memtables 
    // setting can be converted into a function, class constructor is not getting called. 
    debug.print("Active memtable allocated with size: " + std::to_string(activeTableSize)); 
    
    allocateMemoryManaged((void**)&table, sizeof(GMemtable)); 
    allocateAndSetupTable(table, activeTableSize, batchID, keyLength, memtableID);
    *activeTable = table;

    GMemtableLog** gWAL;
    allocateMemoryManaged((void**)&gWAL, sizeof(GMemtableLog) * numMemtablesNeeded);
    allocateMemoryManaged((void**)&gWAL[0], numMemtablesNeeded);
    gWAL[0]->setupLog(fileLocation, table);

    GpuPuts gPuts(gBatch, activeTable, gWAL[0]);
    gPuts.sortPutsOnGPU();

    // SstWriter sstWriter(*activeTable, db, fileLocation); // TODO: move out of the critical path and run in parallel
    debug.print("****************Active table processing done. Pushing to Queue.");
    // Delete gWAL[0]
    pushMemtableToQueue(*activeTable);  
    memtableID++;
    numMemtablesNeeded--; 

    if (numMemtablesNeeded > 1) {
        debug.print("Number of memtables needed is lesser than the number of memtables allowed");
        // allocateMemoryManaged((void**)&immutableTables, sizeof(GMemtable*) * numMemtablesNeeded); // already allocated in batch constructor
        for(int i = 0; i < numMemtablesNeeded; i++) {
            GMemtable* table;
            allocateMemoryManaged((void**)&table, sizeof(GMemtable));
            // For last immutable memtable, separately calculate the size as it could be less than the config size 
            uint64_t immutableMemtableSize = (i == numMemtablesNeeded) ? 
                (writeBatchSize - config.getMemtableSize() * numMemtablesNeeded) : config.getMemtableSize();
            allocateMemoryManaged((void**)&table, sizeof(GMemtable)); 

            allocateAndSetupTable(table, immutableMemtableSize, batchID, keyLength, memtableID);    
            immutableTables[i] = table;
            memtableID++;

            allocateMemoryManaged((void**)&gWAL[i + 1], numMemtablesNeeded);
            gWAL[i + 1]->setupLog(fileLocation, table);

            GpuPuts gPuts(gBatch, &immutableTables[i], gWAL[i]);
            gPuts.sortPutsOnGPU();

            // SstWriter sstWriter(table, db, fileLocation);
            // debug.print("****************SST writer object created");
            pushMemtableToQueue(immutableTables[i]);
            // Delete gWAL[i + 1]
        }
    } 
}
*/

// This function allocates and sets up memtables when the number of memtables needed are more than the ones specified in config
void WriteSubBatch::allocateMultipleMemtablesMoreThanConfig(GpuWriteBatch* gBatch, int numMemtablesNeeded, 
        GMemtable**& immutableTables, int batchID, int keyLength, int& memtableID) {
    
    int numMemtablesLeftToAllocate = numMemtablesNeeded;
    int numMemtablesAllocated = 0;

    // Allocate gWAL for all the memtables
    GMemtableLog** gWAL;
    allocateMemoryManaged((void**)&gWAL, sizeof(GMemtableLog) * numMemtablesNeeded);    

    while(numMemtablesLeftToAllocate > 0) {
    // Allocate the first config.maxMemtables immutable memtables in reverse order going from the last to the first
        int numImmutableMemtables = (numMemtablesLeftToAllocate > config.maxMemtables) ? config.maxMemtables : numMemtablesLeftToAllocate;
        for(int i = numImmutableMemtables - 1; i > 0; i--) {
            uint64_t immutableMemtableSize = std::min(numWrites - numMemtablesAllocated * config.getMemtableSize(), config.getMemtableSize()); 

            timer->startTimer("memtableAlloc", memtableID); 
            GMemtable* table;
            allocateMemoryManaged((void**)&table, sizeof(GMemtable)); 
            allocateAndSetupTable(table, immutableMemtableSize, batchID, keyLength, memtableID);
            immutableTables[i] = table;
            timer->stopTimer("memtableAlloc", memtableID);
            std::cout << "Time taken: " << timer->getTotalTime("memtableAlloc") << "\n";    

            allocateMemoryManaged((void**)&gWAL[numMemtablesAllocated], sizeof(GMemtableLog));
            gWAL[numMemtablesAllocated]->setupLog(fileLocation, immutableTables[i]);
            debug.print("GMemtableLog setup complete for immutable table " + std::to_string(i));
            // std::cout << "GMemtableLog setup complete for immutable table " + numMemtablesAllocated << "\n";

            timer->startTimer("sortPuts", memtableID);
            GpuPuts gPuts(gBatch, &immutableTables[i], gWAL[numMemtablesAllocated]);        
            debug.print("GpuPuts object created for immutable table " + std::to_string(i));
            gPuts.sortPutsOnGPU();
            debug.print("Puts sorted on GPU for immutable table " + std::to_string(i));
            pushMemtableToQueue(immutableTables[i]);
            immutableTables[i] = nullptr; // Set immutable table to null after pushing to queue
            debug.print("Immutable table " + std::to_string(i) + " pushed to queue");
            timer->stopTimer("sortPuts", memtableID);
            std::cout << "Time taken: " << timer->getTotalTime("sortPuts") << "\n";

            debug.print("Immutable table " + std::to_string(i) + " pushed to queue");
            memtableID++;
            numMemtablesAllocated++;
            numMemtablesLeftToAllocate--;
            if (numMemtablesLeftToAllocate == 1) {
                break;
            }
        }
        // Allocate activeTable similarly 
        uint64_t activeTableSize = std::min(numWrites - numMemtablesAllocated * config.getMemtableSize(), config.getMemtableSize());
        GMemtable* table;

        timer->startTimer("memtableAlloc", memtableID);
        allocateMemoryManaged((void**)&table, sizeof(GMemtable));
        allocateAndSetupTable(table, activeTableSize, batchID, keyLength, memtableID);
        *activeTable = table;
        timer->stopTimer("memtableAlloc", memtableID);
        std::cout << "Time taken: " << timer->getTotalTime("memtableAlloc") << "\n";

        allocateMemoryManaged((void**)&gWAL[numMemtablesAllocated], sizeof(GMemtableLog));
        gWAL[numMemtablesAllocated]->setupLog(fileLocation, *activeTable);
        debug.print("GMemtableLog setup complete");
        // std::cout << "GMemtableLog setup complete for active table, memtableID: " << numMemtablesAllocated << "\n"; 

        timer->startTimer("sortPuts", memtableID);
        GpuPuts gPuts(gBatch, activeTable, gWAL[numMemtablesAllocated]);
        debug.print("GpuPuts object created");
        gPuts.sortPutsOnGPU();
        debug.print("Puts sorted on GPU");
        pushMemtableToQueue(*activeTable);
        *activeTable = nullptr; // Set active table to null after pushing to queue
        debug.print("Active table pushed to queue");
        timer->stopTimer("sortPuts", memtableID);
        std::cout << "Time taken: " << timer->getTotalTime("sortPuts") << "\n";

        memtableID++;
        numMemtablesAllocated++;
        numMemtablesLeftToAllocate--;

        // Check if activeTable or immutableTables pointers are not null and more memtables need to be allocated
        // If number of memtables left to allocate is greater than maxMemtables then we need to evict all the memtables
        // Otherwise we need to evict only the required number of memtables
        if (*activeTable != nullptr && numMemtablesLeftToAllocate > 0) {
            if (numMemtablesLeftToAllocate > config.maxMemtables) {
                debug.print("Number of memtables left to allocate is greater than the number of memtables allowed");
                for(int i = 0; i < config.maxMemtables - 1; i++) {
                    if (immutableTables[i] != nullptr) {
                        immutableTables[i]->freeGMemtable();
                    }
                } 
                if (activeTable != nullptr) {
                    (*activeTable)->freeGMemtable(); 
                }
            } else {
                debug.print("Number of memtables left to allocate is lesser than the number of memtables allowed");
                for(int i = 0; i < numMemtablesLeftToAllocate - 1; i++) {
                    if (immutableTables[i] != nullptr) {
                        immutableTables[i]->freeGMemtable(); 
                    }
                }
                if (*activeTable != nullptr) {
                    (*activeTable)->freeGMemtable(); 
                }
            }
        }
    }
    // allocateMultipleMemtablesMoreThanConfig(gBatch, numMemtablesLeftToAllocate, immutableTables, batchID, keyLength, memtableID);
}

void WriteSubBatch::allocateMultipleMemtablesLessThanConfig(GpuWriteBatch* gBatch, int numMemtablesNeeded, 
        GMemtable**& immutableTables, int batchID, int keyLength, int& memtableID) {
    
    int numMemtablesLeftToAllocate = numMemtablesNeeded;
    int numMemtablesAllocated = 0;

    // Allocate gWAL for all the memtables
    GMemtableLog** gWAL;
    allocateMemoryManaged((void**)&gWAL, sizeof(GMemtableLog) * numMemtablesNeeded);

    // Allocate the first numMemtablesNeeded - 1 immutable memtables in reverse order going from the last to the first
    // The last memtable will be the active table
    for(int i = numMemtablesNeeded - 1; i > 0; i--) {
        uint64_t immutableMemtableSize = std::min(numWrites - numMemtablesAllocated * config.getMemtableSize(), config.getMemtableSize()); 

        timer->startTimer("memtableAlloc", memtableID);
        GMemtable* table;
        allocateMemoryManaged((void**)&table, sizeof(GMemtable)); 
        allocateAndSetupTable(table, immutableMemtableSize, batchID, keyLength, memtableID);
        immutableTables[i] = table;
        timer->stopTimer("memtableAlloc", memtableID);
        std::cout << "Time taken: " << timer->getTotalTime("memtableAlloc") << "\n"; 

        allocateMemoryManaged((void**)&gWAL[i], sizeof(GMemtableLog));
        gWAL[i]->setupLog(fileLocation, immutableTables[i]);
        debug.print("GMemtableLog setup complete for immutable table " + std::to_string(i));
        GpuPuts gPuts(gBatch, &immutableTables[i], gWAL[i]);

        timer->startTimer("sortPuts", memtableID);
        debug.print("GpuPuts object created for immutable table " + std::to_string(i));
        gPuts.sortPutsOnGPU();
        debug.print("Puts sorted on GPU for immutable table " + std::to_string(i));
        pushMemtableToQueue(immutableTables[i]);
        immutableTables[i] = nullptr; // Set immutable table to null after pushing to queue
        debug.print("Immutable table " + std::to_string(i) + " pushed to queue");
        timer->stopTimer("sortPuts", memtableID);  
        std::cout << "Time taken: " << timer->getTotalTime("sortPuts") << "\n";

        memtableID++;
        numMemtablesAllocated++;
        numMemtablesLeftToAllocate--;
        if (numMemtablesLeftToAllocate == 1) {
            break;
        }
    }
    // Allocate activeTable similarly
    uint64_t activeTableSize = std::min(numWrites - numMemtablesAllocated * config.getMemtableSize(), config.getMemtableSize());
    
    timer->startTimer("memtableAlloc", memtableID);
    GMemtable* table;
    allocateMemoryManaged((void**)&table, sizeof(GMemtable));
    allocateAndSetupTable(table, activeTableSize, batchID, keyLength, memtableID);
    *activeTable = table;
    timer->stopTimer("memtableAlloc", memtableID);

    allocateMemoryManaged((void**)&gWAL[numMemtablesAllocated], sizeof(GMemtableLog));
    gWAL[numMemtablesAllocated]->setupLog(fileLocation, table);
    debug.print("GMemtableLog setup complete");

    timer->startTimer("sortPuts", memtableID);
    GpuPuts gPuts(gBatch, activeTable, gWAL[numMemtablesAllocated]);
    debug.print("GpuPuts object created");
    gPuts.sortPutsOnGPU();
    debug.print("Puts sorted on GPU");
    pushMemtableToQueue(*activeTable);
    activeTable = nullptr; // Set active table to null after pushing to queue
    debug.print("Active table pushed to queue");
    timer->stopTimer("sortPuts", memtableID);

    memtableID++;
    numMemtablesAllocated++;
    numMemtablesLeftToAllocate--;
}

void WriteSubBatch::gMemtableAllocation() {
    numMemtablesNeeded = (numWrites + config.memtableSize - 1) / config.memtableSize;
    debug.print("Config: " + std::to_string(config.getMemtableSize()) + " Num writes: " + std::to_string(numWrites) + " Num memtables: " + 
        std::to_string(numMemtablesNeeded)); 
    // debug.print("Number of memtables needed: " + std::to_string(numMemtablesNeeded));
    // Allocate active table
    if (numMemtablesNeeded - 1 < config.maxMemtables) {
        allocateMultipleMemtablesLessThanConfig(gBatch, numMemtablesNeeded, immutableTables, batchID, keyLength, memtableID);  
    } else {
        // TODO: Handle this part of the code, we need to divide the write sub-batch further and execute it 
        debug.print("Number of memtables needed is greater than the number of memtables allowed");
        // If the number of memtables needed is greater than the number of memtables allowed, then we have to repeat the process iteratively
       allocateMultipleMemtablesMoreThanConfig(gBatch, numMemtablesNeeded, immutableTables, batchID, keyLength, memtableID); 
    }
}

void WriteSubBatch::gBatchAllocation() {
    timer->startTimer("gBatchAlloc", batchID);

    // Host buffers (pinned)
    hostPinnedKeys = (char*)malloc(sizeof(char) * numWrites * keyLength);
    memcpy(hostPinnedKeys, keys.data(), numWrites * keyLength);
    cudaHostRegister(hostPinnedKeys, numWrites * keyLength, cudaHostRegisterDefault);

    hostPinnedValuesPtr = (char**)malloc(sizeof(char*) * numWrites);
    memcpy(hostPinnedValuesPtr, valuePtrArr, numWrites * sizeof(char*));
    cudaHostRegister(hostPinnedValuesPtr, numWrites * sizeof(char*), cudaHostRegisterDefault);

    hostPinnedOpID = (uint64_t*)malloc(sizeof(uint64_t) * numWrites);
    memcpy(hostPinnedOpID, opIDArr.data(), numWrites * sizeof(uint64_t));
    cudaHostRegister(hostPinnedOpID, numWrites * sizeof(uint64_t), cudaHostRegisterDefault);

    debug.print("Size of keys: " + std::to_string(keys.size()));

    // Device/UM buffers
    allocateMemory((void**)& gBatch->keys,      numWrites * keyLength);
    allocateMemory((void**)& gBatch->valuesPtr, numWrites * sizeof(char*));
    allocateMemory((void**)& gBatch->opID,      numWrites * sizeof(uint64_t));
    debug.print("Allocated memory for keys, valuesPtr and opID in GPU Write Batch");

    // H2D copies
    copyMemoryAsync(gBatch->keys,      hostPinnedKeys,      numWrites * keyLength,          cudaMemcpyHostToDevice);
    copyMemoryAsync(gBatch->valuesPtr, hostPinnedValuesPtr, numWrites * sizeof(char*),      cudaMemcpyHostToDevice);
    copyMemoryAsync(gBatch->opID,      hostPinnedOpID,      numWrites * sizeof(uint64_t),   cudaMemcpyHostToDevice);

    gBatch->numWrites   = numWrites;
    gBatch->keyLength   = keyLength;
    gBatch->valueLength = valueLength;
    gBatch->batchID     = batchID;

    timer->stopTimer("gBatchAlloc", batchID);
}


void WriteSubBatch::gWALAllocation() {
    // May move some code here later
}

void WriteSubBatch::execute() {
    std::cout << "************* EXECUTING THE WRITE SUB-BATCH ON THE GPU ****\n";
    // Determine the number of memtables needed for supporting the batch size 
    memtableConsumerThread = std::thread(&WriteSubBatch::convertMemtableToSST, this); // Launch thread with convertMemtableToSST
    // memtableConsumerThread.detach(); 
    debug.setDebugMode(DEBUG);

    valueArrConversion();

    // TODO: change the fileLocation to what was provided by the user 
    // TODO: run this in parallel with another process

    debug.print("Executing the write sub batch");

    allocateMemoryManaged((void**)&gBatch, sizeof(GpuWriteBatch));  
    
    gBatchAllocation();  
    if(persistThread.joinable()) {
        persistThread.join(); // Wait for persist thread to finish persisting here
    }
    memtableProducerThread = std::thread(&WriteSubBatch::gMemtableAllocation, this); // join in destructor
    if (memtableProducerThread.joinable()) {
        memtableProducerThread.join(); 
    }
    if (memtableConsumerThread.joinable()) {
        memtableConsumerThread.join(); 
    }
}