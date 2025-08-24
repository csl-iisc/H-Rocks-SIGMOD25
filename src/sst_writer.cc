#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <algorithm>
#include "bits/stdc++.h"
#include "helper.cuh"
#include "gmemtable.h"
#include "sst_writer.h"
#include <rocksdb/sst_file_writer.h>

using namespace ROCKSDB_NAMESPACE; 

#define NTHREADS 32
#define TOMBSTONE_MARKER nullptr

SstWriter::SstWriter(GMemtable* gpuTable, rocksdb::DB *db, std::string fileLocation)
    : gpuTable(gpuTable), db(db), fileLocation(fileLocation) {
    sstFileWriter.resize(NTHREADS);
    filePaths.resize(NTHREADS);
    debug.setDebugMode(DEBUG);
    cpuTable = new CMemtable();
    copyGMemtToCPU(gpuTable, cpuTable);
    setupSSTFiles(sstFileWriter);
    // convertToSST(sstFileWriter); // TODO: uncomment this 
}


SstWriter::~SstWriter() {
    gpuTable->freeGMemtable();
    cpuTable->freeCMemtable();
    freeMemory(gpuTable);
    delete cpuTable; 
    // Delete the sst file writers
    for (int i = 0; i < NTHREADS; ++i) {
        delete sstFileWriter[i];
    }
    // Delete files at file path
    for (int i = 0; i < NTHREADS; ++i) {
        remove(filePaths[i].c_str());
    }
}

void SstWriter::copyGMemtToCPU(GMemtable *gpuTable, CMemtable *cpuTable) {
    cpuTable->numKeys = gpuTable->numKeys;
    cpuTable->keyLength = gpuTable->keyLength;
    cpuTable->valueLength = gpuTable->valueLength;
    debug.print("numKeys: " + std::to_string(cpuTable->numKeys) + " keyLength: " + std::to_string(cpuTable->keyLength) + " valueLength: " + std::to_string(cpuTable->valueLength)); 
    cpuTable->keys = new char[gpuTable->numKeys * gpuTable->keyLength];
    cpuTable->valuePointer = new char*[gpuTable->numKeys];  
    // cpuTable->values = new char[gpuTable->numKeys * gpuTable->valueLength]; // Not needed because the value buffer is already on the CPU, part of writeSubBatch
    cpuTable->opID = new uint64_t[gpuTable->numKeys]; 
    cpuTable->batchID = gpuTable->batchID;
    debug.print("Copying GMemtable to CPU");

    copyMemoryAsync(cpuTable->keys, gpuTable->keys, gpuTable->numKeys * gpuTable->keyLength, cudaMemcpyDeviceToHost);
    copyMemoryAsync(cpuTable->valuePointer, gpuTable->valuePointer, gpuTable->numKeys * sizeof(char*), cudaMemcpyDeviceToHost); 
    copyMemoryAsync(cpuTable->opID, gpuTable->opID, gpuTable->numKeys * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    debug.print("Copied GMemtable to CPU");
}


void SstWriter::setupSSTFiles(std::vector<SstFileWriter*>& sstFileWriter) {
    // Code goes here
    Options options;
    options.num_levels = 1;
    options.compaction_style = kCompactionStyleNone;
    options.allow_ingest_behind = true;
    options.write_buffer_size = 1024 * 1024 * 1024;
    options.min_write_buffer_number_to_merge = 10;
    options.level0_file_num_compaction_trigger = 10;
    // Path to where we will write the SST file
    // open multiple sst file writers for parallel writes
    
    for (int i = 0; i < NTHREADS; ++i) {
        sstFileWriter[i] = new SstFileWriter(EnvOptions(), options, options.comparator);
        std::string fileName = "/dev/shm/file" + std::to_string(cpuTable->batchID) + std::to_string(i) + ".sst"; 
        debug.print("Opening SST file: " + fileName);
        filePaths[i] = fileName;
        Status s = sstFileWriter[i]->Open(fileName);
        if (!s.ok()) {
            printf("Error while opening file %s, Error: %s\n", filePaths[i].c_str(),
                    s.ToString().c_str());
        }
    }
    debug.print("SST files opened successfully");
}

void SstWriter::convertToSST(std::vector<SstFileWriter*>& sstFileWriter) {
    // Write all the keys from the CMemtable to the SSTFiles divided by the number of threads
    uint64_t numElemsPerThread = cpuTable->numKeys / NTHREADS;
    IngestExternalFileOptions ifo;
    cudaDeviceSynchronize();
// #pragma omp parallel for num_threads(NTHREADS)
    for (int j = 0; j < NTHREADS; j++) {
        std::string key, key_next;
        for (uint64_t i = j * numElemsPerThread; i < (j + 1) * numElemsPerThread; i++) {
            if (i > cpuTable->numKeys - 1)
                break;
            key.assign(cpuTable->keys + i * cpuTable->keyLength, cpuTable->keys + (i + 1) * cpuTable->keyLength);
            //bCache->invalidate(cache, keys + i * keyLen);
            if (cpuTable->valuePointer[i] == TOMBSTONE_MARKER)
                continue;
            if (i + 1 < cpuTable->numKeys - 1) {
                key_next.assign(cpuTable->keys + (i + 1) * cpuTable->keyLength, cpuTable->keys + (i + 2) * cpuTable->keyLength);
                if (key.compare(key_next) == 0)
                    continue;
            }
            debug.print("Writing key: " + key + " to SST file " + std::to_string(j));
            Status s = sstFileWriter[j]->Put(key, "value"); // TODO: assign real value
            if (!s.ok())
                std::cout << s.ToString() << "\n";
            assert(s.ok());
        }
        Status s = sstFileWriter[j]->Finish();
        assert(s.ok());
        s = db->IngestExternalFile({filePaths[j]}, ifo);
    }
}
