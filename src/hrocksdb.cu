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
    lastBatchTimeStamp = std::chrono::high_resolution_clock::now(); 
    previousRequestRate = 0;
    currentBatchSize = config.batchSize; // Initial batch size set to config.batchSize
    debug.print("Current batch size initialized to " + std::to_string(currentBatchSize));
    numMemtablesAcrossBatches = 0;
}

HRocksDB::~HRocksDB() {
    delete currentBatch;
    freeMemory(activeTable);
    freeMemory(immutableTables);
    debug.print("HRocksDB object deleted");
}

void HRocksDB::HOpen(const std::string fileLocation) {
    // Pass the same resolved path to downstream components
    const std::string path =
        (!fileLocation.empty() && fileLocation.front() == '/')
            ? fileLocation
            : (std::string("/pmem/") + fileLocation);

    debug.print("Opening RocksDB at " + path);
    std::cout << "Opening RocksDB at " << path << std::endl;
    RocksDBOperations rdbOpsTemp;
    rdb = rdbOpsTemp.Open(path);

    currentBatch = new Batch(
        batchID, config.batchSize, config,
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
    } else {
        return; 
    }
}

uint64_t HRocksDB::computeRequestRate(Batch* currentBatch) {
    // uint64_t elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(TIME_NOW - lastBatchTimeStamp).count();
    // uint64_t requestRate = (currentBatch->getTotalOperations() * 1000) / elapsedTime;
    uint64_t elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(TIME_NOW - lastBatchTimeStamp).count();
    uint64_t requestRate = (currentBatch->getTotalOperations() * 1000000) / elapsedTime;  // Adjust factor for microseconds

    debug.print("Request rate is: " + std::to_string(requestRate) + " ops/s");
    std::cout << "Request rate is: " << requestRate << " ops/s" << std::endl;
    lastBatchTimeStamp = TIME_NOW; //Update the time stamp 
    return requestRate;
}

// TODO: there is an assumption here that the current batch size is never going past the config.batchSize 
// Add that check and ensure that the ceiling is maintained 

void HRocksDB::updateBatchSize() {
    // If request rate is high then exponentially increase the batch size 
    uint64_t currentRequestRate = computeRequestRate(currentBatch);
    std::cout << "Current request rate: " << currentRequestRate << " Previous request rate: " << previousRequestRate << std::endl;
    // If request rate is higher than the last request rate then we will increase batch size 
    if (currentRequestRate > previousRequestRate) {
        currentBatchSize = (currentRequestRate > 10 * currentBatchSize) ? 10 * currentBatchSize : currentRequestRate;
    } else if (currentRequestRate < previousRequestRate) {
        currentBatchSize = (currentRequestRate < 2 * currentBatchSize) ? currentBatchSize / 2 : currentRequestRate; 
    } else {
        // request rate is remaining uniform 
        // check if batchSize == currentRequestRate or not 
        // if true then do nothing 
        // else set the batchSize to currentRequestRate?? 
        // keeping the condition like this so that later we can change the factor if needed
        // it is possible to increase the currentBatchSize by a factor of 10 until the currentBatchSize == currentRequestRate 
        // For a uniform request rate this will directly set the batchSize to request rate 
        currentBatchSize = (currentRequestRate > currentBatchSize) ? currentRequestRate : currentBatchSize;
    }
    std::cout << "****************** New batch size: " << currentBatchSize << std::endl;
    previousRequestRate = currentRequestRate; 
}

void HRocksDB::updateBatchSize1() {
// if batch size is less than 250000000 then double the batch size 
    currentBatchSize = (currentBatchSize < 250000000) ? 10 * currentBatchSize : currentBatchSize;
}


// void HRocksDB::updateBatchSize2() {
//     // currentBatchSize = 250000000;
//     // if (currentBatchSize <= 100000000) 
//     //     currentBatchSize = 10 * currentBatchSize;
// }

void HRocksDB::batchLimitReached() {
    if (currentBatch->getTotalOperations() >= currentBatchSize) {
        std::cout << "Batch limit reached: " << currentBatchSize << " batchID: " << batchID <<  std::endl;
        std::cout << "Number of GETs in the batch: " << currentBatch->readBatch->getNumReads() << std::endl;
        std::cout << "Number of PUTs in the batch: " <<  currentBatch->writeBatch->getNumWrites() << std::endl;
        updateBatchSize1(); 
        debug.print("Batch limit reached");
        // commit or exit previous batch : can be done in a new thread or process 
        timer->startTimer("BATCH_COMMIT", batchID);
        currentBatch->commit(); 
        timer->stopTimer("BATCH_COMMIT", batchID);
        delete currentBatch; 
        debug.print("Batch committed and older batch deleted. Starting a new batch.");
        std::cout << "Batch committed and older batch deleted. Starting a new batch." << std::endl;
        // start a new batch 
        batchID++; 
        Batch *_batch = new Batch(batchID, config.batchSize, config, activeTable, immutableTables, rdb, fileLocation, timer, 
        numMemtablesAcrossBatches, memtableBatchMap); 
        currentBatch = _batch; 
        debug.print("New batch started");
    }
}

void HRocksDB::Close() {
    if (currentBatch->getTotalOperations() == 0) 
        return; 
    timer->startTimer("BATCH_COMMIT", batchID);
    currentBatch->commit();
    timer->stopTimer("BATCH_COMMIT", batchID);
    debug.print("Committing the current batch");
    delete currentBatch; 
}

void HRocksDB::Delete(std::string fileLocation) {
    Close(); 
    // Delete the database directory
    std::string pFileLocation = "/pmem" + fileLocation; 
    std::string command = "rm -rf " + pFileLocation;
    // system(command.c_str());
}