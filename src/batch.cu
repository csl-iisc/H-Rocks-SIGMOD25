#include "batch.h"
#include "config.h"
#include "debugger.h"
#include "gmemtable.h"
#include "helper.cuh"
#include "sub_batch.h"

Batch::Batch(int batchID, uint64_t batchSize, Config config,
             GMemtable** activeTable, GMemtable** immutableTables,
             rocksdb::DB* db, std::string fileLocation, DbTimer* timer,
             int& numMemtablesAcrossBatches,
             std::unordered_map<int,int>& memtableBatchMap)
    : batchID(batchID),
      batchSize(batchSize),
      config(config),
      activeTable(activeTable),          // just store the pointers you were given
      immutableTables(immutableTables),  // do NOT reallocate these
      db(db),
      fileLocation(fileLocation),
      timer(timer),
      numMemtablesAcrossBatches(numMemtablesAcrossBatches),
      memtableBatchMap(memtableBatchMap) {

    totalOperations = 0;

    assert(this->activeTable != nullptr && "activeTable slot pointer must be allocated by HRocksDB");
    assert(this->immutableTables != nullptr && "immutableTables array must be allocated by HRocksDB");

    // Optional: initialize slots, but do NOT allocate the slots again.
    if (this->activeTable) {
        // *activeTable is a GMemtable*, the slot. Ensure it's empty to start.
        *this->activeTable = nullptr;  // only set if you know ownership; otherwise omit
    }
    if (this->immutableTables) {
        for (int i = 0; i < config.getNumMemtables(); ++i) {
            this->immutableTables[i] = nullptr;   // array of slots allocated by HRocksDB
        }
    }

    writeBatch = new WriteSubBatch(batchID, batchSize, config,
                                   this->activeTable, this->immutableTables,
                                   db, fileLocation, timer,
                                   numMemtablesAcrossBatches, memtableBatchMap);

    readBatch = new ReadSubBatch(batchID, batchSize, config,
                                 this->activeTable, this->immutableTables,
                                 blockCache, *writeBatch, db, fileLocation, timer);

    updateBatch = new UpdateSubBatch(batchID, batchSize, config,
                                     this->activeTable, this->immutableTables,
                                     blockCache, *writeBatch, db, fileLocation, timer);

    Debugger debug(DEBUG);

}


Batch::~Batch() {
    if (writeBatch) { delete writeBatch; writeBatch = nullptr; }
    if (readBatch)  { delete readBatch;  readBatch  = nullptr; }
    if (updateBatch){ delete updateBatch; updateBatch = nullptr; } // if allocated
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

