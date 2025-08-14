#pragma once
#include <iostream>
#include <stdio.h>

// Default values for memtable size and number of memtables etc. 
// Grow factor tells by how much the batch size grows when the request rate increases 
// Shrink factor tells by how much does the batch size drop when the request rate drops 

#define DEFAULT_MEMTABLE_SIZE 100000000
#define DEFAULT_NUM_MEMTABLES 5
#define DEFAULT_BATCH_SIZE 250000000
#define BLK_CACHE_SIZE 10000000
#define GROW_FACTOR 10
#define SHRINK_FACTOR 2 

#ifdef DEBUG_MODE
    #define DEBUG true
#else
    #define DEBUG false
#endif

// Config class to hold the configuration values
class Config {
public:
    uint64_t memtableSize; // Number of KV pairs allowed in a memtable 
    int maxMemtables; // Number of immutable memtables allowed
    uint64_t batchSize; 
    uint64_t blkCacheSize; 
    int growFactor;
    int shrinkFactor; 

    uint64_t getMemtableSize(); 
    int getNumMemtables();
    uint64_t getBatchSize(); 
    uint64_t getBlkCacheSize(); 
    int getGrowFactor(); 
    int getShrinkFactor(); 
    
    void setMemtableSize(uint64_t size);
    void setNumMemtables(int num);
    void setBatchSize(uint64_t size); 
    void setBlkCacheSize(uint64_t size); 
    void setGrowFactor(int size); 
    void setShrinkFactor(int size); 

    // Constructor with default values
    Config() : memtableSize(DEFAULT_MEMTABLE_SIZE), maxMemtables(DEFAULT_NUM_MEMTABLES), batchSize(DEFAULT_BATCH_SIZE), blkCacheSize(blkCacheSize),
               growFactor(GROW_FACTOR), shrinkFactor(SHRINK_FACTOR) {}
};
