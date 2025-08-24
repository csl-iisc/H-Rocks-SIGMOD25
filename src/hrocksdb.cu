#include "hrocksdb.h" 
#include "config.h"
#include "batch.h"
#include "sub_batch.h"
#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/options.h>
#include "gmemtable.h"
#include "db_timer.h"
#include "rocksdb_ops.h"
#include "helper.cuh"

#define CPU_LIMIT 10000
#define TIME_NOW std::chrono::high_resolution_clock::now()


HRocksDB::HRocksDB(Config config): config(config) {
    batchID = 0; 
    opID = 0; 
    debug.setDebugMode(DEBUG);
    debug.print("HRocksDB object created with batch size " + std::to_string(config.batchSize));
    // create a batch object, set currentBatch to that object 
   
    allocateMemoryManaged((void**)&activeTable, sizeof(GMemtable*));
    debug.print("Active table pointer allocated");
    allocateMemoryManaged((void**)&immutableTables, config.maxMemtables * sizeof(GMemtable*));
    debug.print("Immutable table pointers allocated");
    timer = new DbTimer();
    executingOnCPU = true; 
    lastBatchTimeStamp = TIME_NOW; 
    previousRequestRate = 0;
    currentBatchSize = config.batchSize; // Initial batch size set to config.batchSize
    debug.print("Current batch size initialized to " + std::to_string(currentBatchSize));
    numMemtablesAcrossBatches = 0;
    currentBatchSize = std::max<uint64_t>(CPU_LIMIT, 1);
    maxBatchCap      = currentBatchSize;
}

HRocksDB::~HRocksDB() {
    // Make destruction safe even if Close() was already called
    Close();

    if (rdbOps) { delete rdbOps; rdbOps = nullptr; }
    if (timer)  { delete timer;  timer  = nullptr; }

    if (activeTable)     { freeMemory(activeTable);     activeTable = nullptr; }
    if (immutableTables) { freeMemory(immutableTables); immutableTables = nullptr; }

    debug.print("HRocksDB object deleted");
}


void HRocksDB::HOpen(const std::string fileLocation) {
    // Pass the same resolved path to downstream components
    path =
        (!fileLocation.empty() && fileLocation.front() == '/')
            ? fileLocation
            : (std::string("/pmem/") + fileLocation);

    debug.print("Opening RocksDB at " + path);
    std::cout << "Opening RocksDB at " << path << std::endl;
    RocksDBOperations rdbOpsTemp;
    rdb = rdbOpsTemp.Open(path);

    currentBatch = new Batch(
        batchID, currentBatchSize, config,
        activeTable, immutableTables,
        rdb, path, timer, numMemtablesAcrossBatches, memtableBatchMap);

    rdbOps = new RocksDBOperations(rdb, debug, timer);
}

void HRocksDB::Put(const std::string& key, const std::string& value) {
    currentTimeStamp = TIME_NOW; 
    opID++; 
    debug.print("opID: " + std::to_string(opID));
    if (executingOnCPU) {
        debug.print("Executing on CPU");
        // timer->startTimer("CPU_PUT", batchID);
        executeOnCPU(PUT, key, value);
        // timer->stopTimer("CPU_PUT", batchID);
        return;
    }

    debug.print("Executing on GPU");
    currentBatch->writeBatch->addWriteOperation(key, value, opID);
    debug.print("Put operation added to the write sub batch Inserting: " + key + " : " + value + " Total operations: " + 
        std::to_string(currentBatch->getTotalOperations()) + " Batch ID: " + std::to_string(batchID));
    batchLimitReached(); 
}

void HRocksDB::Delete(const std::string& key) {
    opID++; 
    if (executingOnCPU) {
        executeOnCPU(DELETE, key, NULL);
        return; 
    } 
    currentBatch->writeBatch->addWriteOperation(key, NULL, opID);
    debug.print("Delete operation added to the write sub batch");
    debug.print("Deleting: " + key);
    debug.print("Total operations: " + std::to_string(currentBatch->getTotalOperations()));
    batchLimitReached(); 
}

void HRocksDB::Get(const std::string &key) {
    opID++;
    if (executingOnCPU) {
        executeOnCPU(GET, key, NULL);
        return;
    }
    currentBatch->readBatch->addReadOperation(key, opID);
    debug.print("Get operation added to the read sub batch");
    debug.print("Getting: " + key);
    debug.print("Total operations: " + std::to_string(currentBatch->getTotalOperations()));
    batchLimitReached(); 
}

void HRocksDB::Merge(const std::string &key) {
    currentBatch->updateBatch->addUpdateOperation(key, opID);
    debug.print("Merge operation added to the update sub batch");
    debug.print("Merging: " + key);
    debug.print("Total operations: " + std::to_string(currentBatch->getTotalOperations()));
    opID++;
    batchLimitReached(); 
}

void HRocksDB::Range(const std::string &startKey, const std::string &endKey) {
    currentBatch->rangeBatch->addRangeOperation(startKey, endKey, opID);
    debug.print("Range operation added to the range sub batch");
    debug.print("Range query: " + startKey + " : " + endKey);
    debug.print("Total operations: " + std::to_string(currentBatch->getTotalOperations()));
    opID++;
    batchLimitReached(); 
}

void HRocksDB::executeOnCPU(OperationType type, std::string key, std::string value) {
    if (opID < CPU_LIMIT) {
        if (type == PUT || type == DELETE) {
            rdbOps->Put(const_cast<char*>(key.c_str()), const_cast<char*>(value.c_str()));
        } else if (type == GET) {
            rdbOps->Get(const_cast<char*>(key.c_str()));
        }
    } else if (executingOnCPU && (opID == CPU_LIMIT)) { // First transition from CPU to GPU 
        debug.print("CPU LIMIT REACHED!!");
        if (type == PUT || type == DELETE) {
            rdbOps->Put(const_cast<char*>(key.c_str()), const_cast<char*>(value.c_str()));
        } else if (type == GET) {
            rdbOps->Get(const_cast<char*>(key.c_str()));
        }
        rdbOps->Flush(); 
        opID = 0; 
        executingOnCPU = false; // subsequent operations will execute on the GPU
        lastBatchTimeStamp = TIME_NOW;
        previousRequestRate = 0;
    } else {
        return; 
    }
}

void HRocksDB::updateBatchSizeFromSample(uint64_t ops_in_batch, uint64_t elapsed_us) {
    if (elapsed_us == 0) elapsed_us = 1;
    uint64_t rate = (ops_in_batch * 1'000'000ULL) / elapsed_us;  // ops/sec

    // Desired next-batch size so it can fill within targetFillMs (~1s default)
    uint64_t target_ms   = std::max<uint64_t>(1, 1000);   // e.g., 1000
    uint64_t desired_ops = std::max<uint64_t>(1, (rate * target_ms) / 1000);  // ops

    // Grow/shrink CAP by trend…
    if (rate >= previousRequestRate) {
        // …but don’t stop at one 10× step; keep going until cap covers 'desired_ops'
        int gf = std::max(1, config.getGrowFactor()); // e.g., 10
        while (maxBatchCap < desired_ops && maxBatchCap < config.getBatchSize()) {
            uint64_t next = maxBatchCap > (UINT64_MAX / gf) ? config.getBatchSize()
                                                            : maxBatchCap * gf; // overflow guard
            if (next == maxBatchCap) break;
            maxBatchCap = std::min<uint64_t>(next, config.getBatchSize());
        }
    } else {
        int sf = std::max(1, config.getShrinkFactor()); // e.g., 2
        maxBatchCap = std::max<uint64_t>(maxBatchCap / sf, 10000); // never below 10k ops
    }

    // Actual next batch size: limited by both the cap and the desired fill
    currentBatchSize = std::min<uint64_t>(maxBatchCap, desired_ops);

    std::cout << "[BS policy] rate=" << rate
              << " prev=" << previousRequestRate
              << " cap="  << maxBatchCap
              << " desired=" << desired_ops
              << " -> next batch size=" << currentBatchSize << std::endl;

    previousRequestRate = rate;
}

uint64_t HRocksDB::computeRequestRate(Batch* currentBatch) {
    auto now = TIME_NOW;
    uint64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(now - lastBatchTimeStamp).count();
    if (elapsed_us == 0) elapsed_us = 1; // guard

    // “Incoming rate” here = ops accumulated in current batch / elapsed time
    // between rate samples. This gives ops/sec.
    uint64_t ops = currentBatch->getTotalOperations();
    uint64_t rate = (ops * 1'000'000ULL) / elapsed_us;

    debug.print("Request rate is: " + std::to_string(rate) + " ops/s");
    std::cout << "Request rate is: " << rate << " ops/s" << std::endl;

    lastBatchTimeStamp = now; // start window for the next sample
    return rate;
}

void HRocksDB::updateBatchSize() {
    // 1) Measure
    currentRequestRate = computeRequestRate(currentBatch);

    // 2) Adjust the CAP based on rate trend
    if (currentRequestRate >= previousRequestRate) {
        // steady or increasing → expand by growFactor
        // avoid overflow and clamp to config.batchSize (MAX cap)
        if (maxBatchCap <= config.batchSize / std::max(1, config.growFactor))
            maxBatchCap *= std::max(1, config.growFactor);
        maxBatchCap = std::min<uint64_t>(maxBatchCap, config.batchSize);
    } else {
        // decreasing → shrink by shrinkFactor
        maxBatchCap = std::max<uint64_t>(maxBatchCap / std::max(1, config.shrinkFactor),
                                          CPU_LIMIT);
    }

    // 3) Actual batch cannot exceed incoming rate per second
    uint64_t target = std::min<uint64_t>(
        maxBatchCap,
        (currentRequestRate == 0 ? 1 : currentRequestRate)
    );

    currentBatchSize = std::max<uint64_t>(target, 1ULL);

    std::cout << "[BS policy] rate=" << currentRequestRate
              << " prev=" << previousRequestRate
              << " cap="  << maxBatchCap
              << " -> new batch size=" << currentBatchSize << std::endl;

    previousRequestRate = currentRequestRate;
}

void HRocksDB::batchLimitReached() {
    if (currentBatch->getTotalOperations() >= currentBatchSize) {
      const uint64_t ops_in_batch = currentBatch->getTotalOperations();
      auto now = TIME_NOW;
      uint64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(now - lastBatchTimeStamp).count();
      if (elapsed_us == 0) elapsed_us = 1;
  
      updateBatchSizeFromSample(ops_in_batch, elapsed_us);
      lastBatchTimeStamp = now;
  
      timer->startTimer("BATCH_COMMIT", batchID);
      currentBatch->commit();
      timer->stopTimer("BATCH_COMMIT", batchID);
      delete currentBatch;
  
      // Allocate NEXT batch using the UPDATED size:
      ++batchID;
      std::cout << "[BATCH NEW] id=" << batchID
                << " size=" << currentBatchSize << std::endl;
  
      currentBatch = new Batch(
        batchID, currentBatchSize, config,
        activeTable, immutableTables, rdb, path, timer,
        numMemtablesAcrossBatches, memtableBatchMap);
    }
  }

void HRocksDB::Close() {
    if (currentBatch) {
        if (currentBatch->getTotalOperations() > 0) {
            timer->startTimer("BATCH_COMMIT", batchID);
            currentBatch->commit();
            timer->stopTimer("BATCH_COMMIT", batchID);
            debug.print("Committing the current batch");
        }
        delete currentBatch;
        currentBatch = nullptr;  // <-- critical: prevent double delete
    }
}

void HRocksDB::Delete(std::string fileLocation) {
    Close(); 
    // Delete the database directory
    std::string pFileLocation = "/pmem" + fileLocation; 
    std::string command = "rm -rf " + pFileLocation;
    // system(command.c_str());
}