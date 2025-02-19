#include "gpu_gets.cuh"
#include <string.h>

#define BUFFER_FRACTION 0.5
GpuGets::GpuGets(ReadSubBatch* readBatch, GMemtable** activeTable, GMemtable** immutableMemtable, BlockCache* blockCache, 
        SharedBuffer* sharedBuffer, DbTimer* timer) :  readBatch(readBatch), activeTable(activeTable), immutableTable(immutableTable), 
        blockCache(blockCache), sharedBuffer(sharedBuffer), timer(timer) {

    Debugger debug(DEBUG); 
    
    memtableHits = 0; 
    memtableMisses = 0; 

    blockCacheHits = 0; 
    blockCahceMiss = 0; 
  
    numKeys = readBatch->numReads; // in the read sub-batch 
    keyLength = readBatch->keyLength; 
    debug.print("Number of keys in the read sub-batch: " + std::to_string(numKeys));
    allocateMemory((void**)&gValuePointersArr, numKeys * sizeof(char*)); 
    
    // Key not found buffer is used to store the keys that are not found in the memtables or the block cache 
    // Size of the buffer is set to half of the number of keys in the read sub-batch 
}

GpuGets::~GpuGets() {
    freeMemory(gValuePointersArr);
}