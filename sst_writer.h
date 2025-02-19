#include <rocksdb/sst_file_writer.h>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include "gmemtable.h"
#include "debugger.h"

using namespace ROCKSDB_NAMESPACE;

class SstWriter {
    GMemtable* gpuTable;
    rocksdb::DB *db;
    CMemtable* cpuTable; 
    std::string fileLocation;
    std::vector<SstFileWriter*> sstFileWriter; 
    std::vector<std::string> filePaths;
    Debugger debug; 

    public:
    SstWriter(GMemtable* gpuTable, rocksdb::DB *db, std::string fileLocation); 
    ~SstWriter();
    void convertToSST(std::vector<SstFileWriter*>& sstFileWriter); 
    void setupSSTFiles(std::vector<SstFileWriter*>& sstFileWriter); 
    void copyGMemtToCPU(GMemtable* gpuTable, CMemtable* cpuTable); 

}; 