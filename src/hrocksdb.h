#pragma once

#include "batch.h"
#include "config.h"
#include "debugger.h"
#include <rocksdb/db.h>
#include "gmemtable.h"
#include "db_timer.h"
#include "rocksdb_ops.h"
#include <chrono>

class HRocksDB {

public:
    HRocksDB(Config config); 
    ~HRocksDB();    
    
    void Get(const std::string& key);
    void Put(const std::string& key, const std::string& value);
    void Delete(const std::string& key);
    void Range(const std::string& startKey, const std::string& endKey);
    void Merge(const std::string& key);    
    void Close(); 
    void HOpen(std::string fileLocation); 
    void Delete(std::string fileLocation);
    void executeOnCPU(OperationType type, std::string key, std::string value);
    uint64_t computeRequestRate(Batch* batch); 
    void updateBatchSize();
    void updateBatchSizeFromSample(uint64_t ops_in_batch, uint64_t elapsed_us);


    GMemtable** activeTable = nullptr; 
    GMemtable** immutableTables = nullptr;  
    DbTimer* timer = nullptr;
    std::string path;


    Batch* currentBatch = nullptr;
    int keyLength = 0;
    int valueLength = 0;
    int batchID = -1;  
    void batchLimitReached();
    Config config;
    Debugger debug;
    rocksdb::DB* rdb = nullptr;
    uint64_t opID;
    std::string fileLocation; 
    RocksDBOperations* rdbOps = nullptr;
    std::unordered_map<int, int> memtableBatchMap;
    
    unsigned int previousRequestRate; 
    unsigned long long int currentBatchSize; 
    int numMemtablesAcrossBatches;
    bool executingOnCPU;
    std::chrono::high_resolution_clock::time_point lastBatchTimeStamp; // Ensure correct type
    std::chrono::high_resolution_clock::time_point currentTimeStamp; 
    uint64_t maxBatchCap = 0;         // dynamic cap that grows/shrinks
    uint64_t currentRequestRate = 0;  // ops/sec

};