#pragma once
#include <string>
#include "gmemtable.h"

#define NUM_TRANSFER_THREADS 8

class GMemtableLog {
    
    uint64_t keySize; 
    uint64_t opIDSize; 
    uint64_t valuePtrSize;
    uint64_t numKeys; 
    int batchID; 
    uint64_t valueSize; 
    int memtableID; 
    GMemtable* gMemt;
    std::string folderName;
    Debugger debug; 

    public: 

    uint64_t* opID; // updated by GPU 
    char* keys; // updated by GPU 
    char** valuePtrs; // updated by GPU
    char* values; // persisted and written by CPU

    void setupLog(std::string folderName, GMemtable* gMemt); 
    void persistValues(std::string folderName, uint64_t batchID, uint64_t numWrites, int valueLength, char* volatile_values);
    void persist(); 
    GMemtableLog();
}; 