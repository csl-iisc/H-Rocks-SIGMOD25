#include "hrocksdb.h"
#include <stdio.h>
#include "config.h"


int main()
{
    Config conf; 
    HRocksDB* hdb = new HRocksDB(conf); // Initialize the db variable

    // rocksdb::DB* rdb;

    // std::string fileLocation = "/pmem/test/";
    // rocksdb::Options options;
    // options.IncreaseParallelism(128);
    // options.compression = rocksdb::CompressionType::kSnappyCompression;
    // options.create_if_missing = true;
    // rocksdb::Status status = rocksdb::DB::Open(options, fileLocation, &rdb);
    // assert(status.ok());    

    hdb->HOpen("test");
    hdb->Put("key1", "value1");
    hdb->Put("key2", "value2");
    // hdb->Get("key1");
    hdb->Close();
    delete(hdb);
}