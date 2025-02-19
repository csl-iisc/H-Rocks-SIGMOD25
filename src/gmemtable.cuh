#pragma once
#include <iostream>

struct GMemtable {
    char* keys;
    char** valuePointer;
    int batchID;
    uint64_t* opID;
    unsigned int keyLength; 
    unsigned int valueLength; 
    uint64_t numKeys;
    uint64_t size; 
    int memtableID;
    int numImmutableMemtables;

    void freeGMemtable(); 
};

typedef struct GMemtable GMemtable; 


struct CMemtable {
    char* keys;
    char** valuePointer;
    int batchID;
    uint64_t* opID;
    unsigned int keyLength; 
    unsigned int valueLength; 
    uint64_t numKeys;
    uint64_t size; 
    int memtableID;

    void freeCMemtable();
};

typedef struct CMemtable CMemtable; 