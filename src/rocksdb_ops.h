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
    UPDATE,
    RANGE,
    FLUSH, 
    OPEN,
    CLOSE
};

class RocksDBOperations {
    rocksdb::DB* db;
    Debugger debug; 
    DbTimer* timer;

public:

    RocksDBOperations() : db(nullptr), debug(false), timer(nullptr) {}
    rocksdb::DB* Open(std::string fileLocation); 
    void Put(char* key, char* value);
    void Get(char* key);
    void Delete(char* key);
    void Flush();
    void Update(char* key, char* value);
    void Range(char* startKey, char* endKey);
    void Merge(char* key);
    RocksDBOperations(rocksdb::DB* db, Debugger debug, DbTimer* timer);
}; 