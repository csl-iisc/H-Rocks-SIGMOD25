// Persist the values in a file 
#include <iostream>

#include "omp.h"
#include "helper.cuh"
#include "debugger.h"
#include "gpu_batch.cuh"
#include "log.cuh"
#include "libgpm.cuh"


#define NUM_TRANSFER_THREADS 8

GMemtableLog::GMemtableLog() {
    Debugger debug(DEBUG);
}

// Set all logs for keys, valuePtrs, opID 
void GMemtableLog::setupLog(std::string folderName, GMemtable* gMemt) {
    debug.setDebugMode(DEBUG);
    
    uint64_t numWrites = gMemt->size; 
    
    valuePtrSize = sizeof(uint64_t) * numWrites; 
    keySize = gMemt->keyLength * numWrites; 
    memtableID = gMemt->memtableID; 
    batchID = gMemt->batchID;

    opIDSize = sizeof(uint64_t) * numWrites; 
    debug.print("opIDSize: " + std::to_string(opIDSize));

    std::string cleanFolder = folderName;
    if (cleanFolder.rfind("/pmem/", 0) == 0) {
        cleanFolder = cleanFolder.substr(6); // drop leading "/pmem"
    }

    // Ensure trailing slash
    if (!cleanFolder.empty() && cleanFolder.back() != '/')
        cleanFolder += '/';
    std::string opIDFileName = cleanFolder + "/opID_" + std::to_string(batchID) + "_" + std::to_string(memtableID) + ".dat"; 
    debug.print("opIDFileName: " + opIDFileName); 
    std::cout << "opIDFileName: " << opIDFileName << std::endl;

    opID = (uint64_t*) gpm_map_file(opIDFileName.c_str(), opIDSize, true);
    
    std::string valuePtrFileName = cleanFolder + "/valuePtr_" + std::to_string(batchID) + "_" + std::to_string(memtableID) + ".dat";
    debug.print("Value Log File Name: " + valuePtrFileName);
    std::cout << "Value Log File Name: " << valuePtrFileName << std::endl;
    
    valuePtrs = (char**) gpm_map_file(valuePtrFileName.c_str(), valuePtrSize, true); 

    std::string keyLogFileName = cleanFolder + "/key_" + std::to_string(batchID) + "_" + std::to_string(memtableID) + ".dat";
    debug.print("Key Log File Name: " + keyLogFileName);
    std::cout << "Key Log File Name: " << keyLogFileName << std::endl;

    keys = (char*) gpm_map_file(keyLogFileName.c_str(), keySize, true);
}  


void GMemtableLog::persistValues(std::string folderName, uint64_t batchID, uint64_t numWrites, int valueLength, char* volatileValues) {

    // If folderName accidentally starts with "/pmem/", strip it
    std::string cleanFolder = folderName;
    if (cleanFolder.rfind("/pmem/", 0) == 0) {
        cleanFolder = cleanFolder.substr(6); // drop leading "/pmem"
    }

    // Ensure trailing slash
    if (!cleanFolder.empty() && cleanFolder.back() != '/')
        cleanFolder += '/';

    // Build final path (gpm_map_file will add "/pmem" itself)
    std::string valuePath = cleanFolder + "values" + std::to_string(batchID) + ".dat";

    debug.print("valuePath: " + valuePath); 
    valueSize = numWrites * valueLength; 
    debug.print("valueSize: " + std::to_string(valueSize)); 
    // values = (char*) gpm_map_file(valuePath.c_str(), valueSize, true);
    values = (char*) gpm_map_file(valuePath.c_str(), valueSize, true);

    uint64_t copySizePerThread = valueSize/NUM_TRANSFER_THREADS; 

#pragma omp parallel for num_threads(NUM_TRANSFER_THREADS)
    for(int i = 0; i < NUM_TRANSFER_THREADS; i++) {
        memcpy(values + i * copySizePerThread, volatileValues + i * copySizePerThread, copySizePerThread);
    }
    debug.print("Values copied to log file.");
    pmem_mt_persist(values, valueSize); 
    debug.print("Values persisted to log file.");
}

void GMemtableLog::persist() {
    pmem_mt_persist(opID, opIDSize); 
    pmem_mt_persist(valuePtrs, valuePtrSize); 
    pmem_mt_persist(keys, keySize); 
}