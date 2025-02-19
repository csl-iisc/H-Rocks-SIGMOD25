#include "sub_batch.h"
#include "debugger.h"
#include "helper.cuh"
#include "db_timer.h"

class GpuGets {
    ReadSubBatch* readBatch; 
    GMemtable** activeTable; 
    GMemtable** immutableTable; 

    uint64_t memtableHits; 
    uint64_t memtableMisses; 

    uint64_t blockCacheHits; 
    uint64_t blockCahceMiss; 

    uint64_t numKeys; 
    int keyLength; 

    Debugger debug; 

    char** gValuePointersArr; 
    GpuReadBatch* gReadBatch;

    BlockCache* blockCache;

    SharedBuffer* sharedBuffer; 

    DbTimer* timer;

    public: 
    GpuGets(ReadSubBatch* readBatch, GMemtable** activeTable, GMemtable** immutableMemtable, BlockCache* blockCache, SharedBuffer* sharedBuffer, DbTimer* timer); 
    ~GpuGets(); 
    void search(); 
};