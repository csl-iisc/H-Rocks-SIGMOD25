#pragma once

#include "sub_batch.h"
#include "config.h"
#include "gmemtable.h"
#include "block_cache.h"
#include "db_timer.h"

class Batch {

private: 
    uint64_t totalOperations; // Total number of operations in the batch
    Config config;
    Debugger debug;
    GMemtable** activeTable;
    GMemtable** immutableTables;
    BlockCache* blockCache;
    rocksdb::DB* db; 
    std::string fileLocation;
    DbTimer* timer;
    int numMemtablesAcrossBatches; // Total number of memtables across all batches
    std::unordered_map<int, int>& memtableBatchMap; // Map to track which batch a memtable belongs to


public:
    Batch(int batchID, uint64_t batchSize, Config config, GMemtable** activeTable, GMemtable** immutableTables, 
        rocksdb::DB* db, std::string fileLocation, DbTimer* timer); 
    Batch(int batchID, uint64_t batchSize, Config config, GMemtable** activeTable, GMemtable** immutableTables, 
        rocksdb::DB* db, std::string fileLocation, DbTimer* timer, int& numMemtablesAcrossBatches, 
        std::unordered_map<int, int>& memtableBatchMap); 
    ~Batch();
    WriteSubBatch* writeBatch;
    ReadSubBatch* readBatch;
    UpdateSubBatch* updateBatch; 
    RangeSubBatch* rangeBatch;
    int batchID; 
    uint64_t batchSize; // max batchSize 
    uint64_t getTotalOperations(); 
    void commit(); 
};
