#pragma once

#include "batch.h"
#include "config.h"
#include "debugger.h"
#include <rocksdb/db.h>
#include "gmemtable.h"
#include "db_timer.h"
#include "rocksdb_ops.h"

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

    GMemtable** activeTable; 
    GMemtable** immutableTables; 
    DbTimer* timer;

    Batch* currentBatch;
    int keyLength;
    int valueLength;
    int batchID; 
    void batchLimitReached();
    Config config;
    Debugger debug;
    rocksdb::DB* rdb;
    uint64_t opID;
    std::string fileLocation; 
    RocksDBOperations* rdbOps; 
};
