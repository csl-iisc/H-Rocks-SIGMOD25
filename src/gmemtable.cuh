#pragma once
#include <iostream>

struct GMemtable {
    char* keys = nullptr; // Contiguous buffer of keys
    char** valuePointer = nullptr; // Array of pointers to values
    int batchID = -1; // ID of the batch this memtable belongs to
    uint64_t* opID = nullptr; // Array of operation IDs
    unsigned int keyLength = 0; 
    unsigned int valueLength = 0; 
    uint64_t numKeys = 0;
    uint64_t size = 0; 
    int memtableID = -1;
    int numImmutableMemtables = 0;

    void freeGMemtable(); 
};

typedef struct GMemtable GMemtable; 


struct CMemtable {
    char* keys = nullptr; // Contiguous buffer of keys
    char** valuePointer = nullptr; // Array of pointers to values
    int batchID = -1; // ID of the batch this memtable belongs to
    uint64_t* opID = nullptr; // Array of operation IDs
    unsigned int keyLength = 0; 
    unsigned int valueLength = 0; 
    uint64_t numKeys = 0;
    uint64_t size = 0; 
    int memtableID = -1;

    void freeCMemtable();
};

typedef struct CMemtable CMemtable; 

static inline void deleteMemtable(GMemtable*& mt); 