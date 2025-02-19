#include "batch.h"
#include "config.h"
#include "debugger.h"
#include "gmemtable.h"
#include "helper.cuh"
#include "sub_batch.h"

Batch::Batch(int batchID, uint64_t batchSize, Config config, GMemtable** activeTable, GMemtable** immutableTables, 
        rocksdb::DB* db, std::string fileLocation, DbTimer* timer, int& numMemtablesAcrossBatches, 
        std::unordered_map<int, int>& memtableBatchMap) 
        : batchID(batchID), totalOperations(0), batchSize(batchSize), config(config), activeTable(activeTable), 
        immutableTables(immutableTables), db(db), fileLocation(fileLocation), timer(timer), 
        numMemtablesAcrossBatches(numMemtablesAcrossBatches), memtableBatchMap(memtableBatchMap){
    allocateMemoryManaged((void**)&activeTable, sizeof(GMemtable*));
    allocateMemoryManaged((void**)&immutableTables, config.getNumMemtables() * sizeof(GMemtable*)); 
    // set the active table and immutable tables to nullptr for now
    // activeTable[0] = nullptr;
    // for (int i = 0; i < config.getNumMemtables(); i++) {
    //     immutableTables[i] = nullptr;
    // }

    writeBatch = new WriteSubBatch(batchID, batchSize, config, activeTable, immutableTables, db, fileLocation, timer, 
        numMemtablesAcrossBatches, memtableBatchMap);
    readBatch = new ReadSubBatch(batchID, batchSize, config, activeTable, immutableTables, blockCache, *writeBatch, db, fileLocation, timer);
    updateBatch = new UpdateSubBatch(batchID, batchSize, config, activeTable, immutableTables, blockCache, *writeBatch, db, fileLocation, timer);
    Debugger debug(DEBUG);
}

Batch::~Batch() {
    delete writeBatch;
    delete readBatch;
    // delete updateBatch;
    // reset others
    batchID = -1; 
    batchSize = 0;
}

uint64_t Batch::getTotalOperations() {
    totalOperations = writeBatch->getNumWrites() + readBatch->getNumReads(); // + updateBatch->getNumUpdates();  
    timer->numOps = totalOperations;
    return totalOperations;
}

// Write the commit function now
void Batch::commit() {
    // Execute the write sub batch 
    debug.setDebugMode(DEBUG);
    debug.print("Committing the write sub batch");
    // Once puts are processed then getNumWrites will not be zero
    if (writeBatch->getNumWrites() > 0) {
        writeBatch->execute();
    }
    // writeBatch->execute();
    cudaDeviceSynchronize();
    if (readBatch->getNumReads() > 0) {
        debug.print("Committing the read sub batch");
        std::cout << "Committing the read sub batch with numReads: " << readBatch->getNumReads() << "\n";
        readBatch->execute();
    }
    if (updateBatch->getNumUpdates() > 0) {
        debug.print("Committing the update sub batch");
        updateBatch->execute();
    }
    // std::cout << "Total time: " << timer->getTotalTime() << "\n";
}