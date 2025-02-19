// Define a class CpuGets which is similar to GpuGets
// but uses the CPU to compute the result.
#pragma once

#include <iostream>
#include "rocksdb/db.h"
#include "block_cache.h"
#include "sub_batch.h"
#include "debugger.h"
#include "db_timer.h"

using namespace rocksdb;


class CpuGets {
    BlockCache* blockCache; 
    SharedBuffer* sharedBuffer;
    rocksdb::DB* db;
    DbTimer* timer;
    Debugger debug;
    bool* _notFoundKeysBitMap; 
    std::vector<char> allKeysToGet; // Contains all keys from the read sub batch to be read
    int keyLength; 
    bool* gpuProcessingDone;
    std::condition_variable* cv;
    
    public:
    CpuGets(std::vector<char> allKeysToGet, int keyLength, rocksdb::DB* db, SharedBuffer* sharedBuffer, BlockCache* cache, 
    bool& gpuProcessingDone, std::condition_variable& cv);
    ~CpuGets();
    void performGets();   
};

// changed this implementation for sharing data between CPU and GPU 
// Ignore 

class CpuGets2 { 
    char** valuePtrs;
    rocksdb::DB *db; 
    BlockCache* blockCache; 
    SharedBuffer2* sharedBuffer2;
    Debugger debug; 
    int keyLength; 
    std::mutex* mtx;
    bool* gpuProcessingDone;
    std::condition_variable* cv;
    DbTimer* timer;

    public:
    CpuGets2(rocksdb::DB* db, SharedBuffer2* sharedBuffer2, char** valuePtrArr, BlockCache* cache, 
        std::mutex& mtx, bool& gpuProcessingDone, std::condition_variable& cv); 
    ~CpuGets2();
    void performGets();    
}; 

