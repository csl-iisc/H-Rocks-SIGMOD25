#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include "config.h"
#include "debugger.h"
#include "gmemtable.h"
#include "gpu_batch.cuh"
#include "log.cuh"
#include "block_cache.h"
#include "rocksdb/db.h"
#include <thread>
#include <queue>    
#include <mutex>
#include <condition_variable>
#include "db_timer.h"

using namespace ROCKSDB_NAMESPACE;

class WriteSubBatch {
    Debugger debug;

    void memtableEviction(GMemtable* table, GMemtableLog* gWAL); 
    void allocateMultipleMemtablesMoreThanConfig(GpuWriteBatch* gBatch, int numMemtablesNeeded, GMemtable**& immutableTables, int batchID, int keyLength, int& memtableID);
    void allocateMultipleMemtablesLessThanConfig(GpuWriteBatch* gBatch, int numMemtablesNeeded, GMemtable**& immutableTables, int batchID, int keyLength, int& memtableID);
    void valueArrConversion();
    void gMemtableAllocation(); 
    void gBatchAllocation(); 
    void gWALAllocation(); 
    int numMemtablesNeeded; 
    void allocateAndSetupTable(GMemtable* table, uint64_t tableSize, uint64_t batchID, uint64_t keyLength, uint64_t memtableID);
    
    // For SST File management in parallel 
    std::thread persistThread;
    std::thread memtableProducerThread; // this thread pushes the memtables to be converted into sstFiles into a queue
    std::thread memtableConsumerThread; // this thread consumes it from the queue and writes it to NVM

    std::queue<GMemtable*> memtableQueue;
    std::mutex queueMutex;
    std::condition_variable queueCondVar;
    void pushMemtableToQueue(GMemtable* table); 
    void convertMemtableToSST();
    DbTimer* timer; 

    int numMemtablesAcrossBatches;
    std::unordered_map<int, int>& memtableBatchMap;
    // write_sub_batch.h (or wherever the class is declared)
    char*     hostPinnedKeys      = nullptr;   // formerly gBatch->cKeys local
    char**    hostPinnedValuesPtr = nullptr;   // formerly _valuesPtr local
    uint64_t* hostPinnedOpID      = nullptr;   // formerly _opID  local


    public: 
    std::vector<char> keys;
    std::vector<char> values;
    std::vector<uint64_t> opIDArr;
    char* valueArr; 

    char** valuePtrArr;
    int keyLength; // Assuming all keys are of the same length
    int valueLength; // Assuming all values are of the same length

    int batchID;
    int memtableID; 

    uint64_t numWrites;
    uint64_t writeBatchSize; 
    Config config;

    GMemtable** activeTable;
    GMemtable** immutableTables;
    GpuWriteBatch* gBatch;
    GMemtableLog* gLog;

    rocksdb::DB* db;
    std::string fileLocation; 

    WriteSubBatch(int batchID, uint64_t writeBatchSize, Config config, GMemtable **activeTable, GMemtable **immutableTables, 
        rocksdb::DB* db, std::string fileLocation, DbTimer* timer, int& numMemtablesAcrossBatches, 
        std::unordered_map<int, int>& memtableBatchMap); 
    WriteSubBatch(int batchID, uint64_t writeBatchSize, Config config, GMemtable **activeTable, GMemtable **immutableTables, 
        rocksdb::DB* db, std::string fileLocation, DbTimer* timer);
    ~WriteSubBatch(); 
    WriteSubBatch();
    void addWriteOperation(std::string key, std::string value, uint64_t opID);   
    uint64_t getNumWrites(); 
    int getValueLength(); 
    int getKeyLength();
    void execute();
    char** getValues(); 

};

// Shared buffer for keys not found in the GPU memtables and cache. This data is maintained in the CPU memory. 
// It is only used for read operations.
struct SharedBuffer2{
    char* notFoundKeysBuffer;
    uint64_t* numNotFoundKeys; // This is the number of keys not found in the GPU memtables and cache 
    uint64_t* head;
    uint64_t* tail; 
    int keyLength;  
    int batchID; 
    uint64_t bufferSize; // This is the capacity of the buffer 
}; 

typedef struct SharedBuffer2 SharedBuffer2; 

struct SharedBuffer {
    uint64_t readBatchSize; 
    // TOOD: improve the implementation to make it bitmap instead although that would require bit wise operations 
    bool* notFoundKeysBitMap; // Array of keys that were not found by the GPU. It is actually a byte map for now
    uint64_t* numNotFoundKeys; 
    int* doorbell; // This is used to signal the CPU that the GPU has finished processing the keys not found in the memtables and cache
}; 

typedef struct SharedBuffer SharedBuffer;


struct NotFoundBuffer {
    uint64_t numReads; 
    bool* notFoundKeysBitMap; 
    unsigned int* numNotFoundKeys;
}; 

typedef struct NotFoundBuffer NotFoundBuffer;

class ReadSubBatch {
public:

    Debugger debug; 
    std::vector<char> keys;
    std::vector<char> values;
    std::vector<uint64_t> opIDArr; 

    char** valuePtrArr; // This has the output for the keys 
    uint64_t numReads; 
    uint64_t readBatchSize; 

    GMemtable** activeTable;
    GMemtable** immutableTables;

    int batchID;

    int keyLength; // Assuming all keys are of the same length
    int valueLength; // Assuming all values are of the same length
    Config config;

    BlockCache* cache;
    WriteSubBatch* writeBatch;

    rocksdb::DB* db;
    std::string fileLocation;

    // For keys not found in the GPU memtables and cache. This data is maintained in the CPU memory. 
    std::mutex mtx;
    std::condition_variable cv;
    bool gpuProcessingDone = false;
    void signalGpuProcessingComplete(); 
    // void allocateSharedBuffer2(); 

    SharedBuffer* sharedBuffer; 
    NotFoundBuffer* notFoundBuffer;
    // SharedBuffer2* sharedBuffer2; // Not needed any more, logic changed 
    std::thread keyNotFoundProducerThread; // Run the GPU kernel to find the keys that are not found in the memtables and cache 
    std::thread keyNotFoundConsumerThread;  // Run the CPU function to lookup keys not found 
    void allocateSharedBuffer(); 

    DbTimer* timer;

    // Constructor
    ReadSubBatch(int batchID, uint64_t readBatchSize, Config config, GMemtable** activeTable, GMemtable** immutableTables, 
        BlockCache* cache, WriteSubBatch& writeBatch, rocksdb::DB* db, std::string fileLocation, DbTimer* timer); 
    ~ReadSubBatch(); 
    void addReadOperation(std::string key, uint64_t opID); 
    uint64_t getNumReads(); 
    void execute();
};


class UpdateSubBatch {
    public:
    std::vector<char> keys;
    std::vector<uint64_t> opIDArr; 
    uint64_t numUpdates;
    uint64_t updateBatchSize; 
    GpuUpdateBatch* gBatch; 

    int batchID;
    int memtableID;
    
    Config config;

    GMemtable** activeTable;
    GMemtable** immutableTables;
    Debugger debug;

    rocksdb::DB* db;
    std::string fileLocation;    
    BlockCache* cache;
    WriteSubBatch* writeBatch;

    int keyLength; // Assuming all keys are of the same length

    std::thread persistThread;
    std::thread memtableProducerThread; // this thread pushes the memtables to be converted into sstFiles into a queue
    std::thread memtableConsumerThread; // this thread consumes it from the queue and writes it to NVM


    void memtableEviction(GMemtable* table, GMemtableLog* gWAL); 
    void allocateMultipleMemtablesMoreThanConfig(GpuWriteBatch* gBatch, int numMemtablesNeeded, GMemtable**& immutableTables, int batchID, int keyLength, int& memtableID);
    void allocateMultipleMemtablesLessThanConfig(GpuWriteBatch* gBatch, int numMemtablesNeeded, GMemtable**& immutableTables, int batchID, int keyLength, int& memtableID);
    void valueArrConversion();
    void gMemtableAllocation(); 
    void gBatchAllocation(); 
    void gWALAllocation(); 
    int numMemtablesNeeded; 
    void allocateAndSetupTable(GMemtable* table, uint64_t tableSize, uint64_t batchID, uint64_t keyLength, uint64_t memtableID);
    
    std::queue<GMemtable*> memtableQueue;
    std::mutex queueMutex;
    std::condition_variable queueCondVar;
    void pushMemtableToQueue(GMemtable* table); 
    void convertMemtableToSST();
    DbTimer* timer; 


    // Constructor
    UpdateSubBatch(int batchID, uint64_t updateBatchSize, Config config, GMemtable** activeTable, GMemtable** immutableTables, 
    BlockCache* cache, WriteSubBatch& WriteBatch, rocksdb::DB* db, std::string fileLocation, DbTimer* timer);
    UpdateSubBatch(int batchID, uint64_t updateBatchSize, Config config, GMemtable** activeTable, GMemtable** immutableTables);
         
    ~UpdateSubBatch();
    void addUpdateOperation(std::string key, uint64_t opID);
    uint64_t getNumUpdates();
    void execute();
};

class RangeSubBatch {
    public:
    std::vector<char> startKeys;
    std::vector<char> endKeys;
    std::vector<uint64_t> opIDArr; 

    uint64_t numRangeQueries;
    uint64_t rangeBatchSize; 
    Config config;
    int batchID;

    int keyLength; // Assuming all keys are of the same length

    // Constructor
    RangeSubBatch(int batchID, uint64_t rangeBatchSize, Config config); 
    ~RangeSubBatch();
    void addRangeOperation(std::string startKey, std::string endKey, uint64_t opID);
    uint64_t getNumRangeQueries();
};