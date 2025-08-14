#pragma once 
#include <iostream>
#include "rocksdb/db.h"
#include "db_timer.h"
#include "debugger.h"

using namespace rocksdb; 

enum OperationType {
    PUT,
    GET,
    DELETE,
    UPDATE
};

class RocksDBOperations {
    rocksdb::DB* db;
    Debugger debug; 
    DbTimer* timer;

public:

    void Put(char* key, char* value);
    void Get(char* key);
    void Delete(char* key);
    void Flush();
    void Update(char* key, char* value);
    void Range(char* startKey, char* endKey);
    void Merge(char* key);
    RocksDBOperations(rocksdb::DB* db, Debugger debug, DbTimer* timer);
}; 