#include "rocksdb_ops.h" 
#include "debugger.h"
#include "db_timer.h"

RocksDBOperations::RocksDBOperations(rocksdb::DB* db, Debugger debug, DbTimer* timer): db(db), debug(debug), timer(timer) {}

rocksdb::DB* RocksDBOperations::Open(std::string fileLocation) {
        rocksdb::Options options;
    options.IncreaseParallelism(128);
    options.OptimizeLevelStyleCompaction();
    options.create_if_missing = true;

    rocksdb::DB* rdb = nullptr;

    rocksdb::Status status = rocksdb::DB::Open(options, fileLocation, &rdb);
    debug.print(status.ToString());
    assert(status.ok());
    debug.print("RocksDB opened successfully");
    return rdb;
}

void RocksDBOperations::Put(char* key, char* value) {
    debug.print("Put executed by CPU");
    // std::cout << "put executed on CPU\n";
    db->Put(WriteOptions(), Slice(key), Slice(value));
}

void RocksDBOperations::Get(char* key) {
    std::string value;
    debug.print("Get executed by CPU");
    Status s = db->Get(ReadOptions(), Slice(key), &value);
    if (s.ok()) {
        std::cout << value << std::endl;
    } else {
        std::cout << "Key not found" << std::endl;
    }
}

void RocksDBOperations::Delete(char* key) {
    debug.print("Delete executed by CPU");
    db->Delete(WriteOptions(), Slice(key));
}

void RocksDBOperations::Flush() {
    debug.print("Flush executed by CPU");
    db->Flush(FlushOptions());
}

void RocksDBOperations::Update(char* key, char* value) {
    debug.print("Update executed by CPU");
    db->Put(WriteOptions(), Slice(key), Slice(value));
}

void RocksDBOperations::Range(char* startKey, char* endKey) {
    debug.print("Range executed by CPU");
    rocksdb::Iterator* it = db->NewIterator(ReadOptions());
    for (it->Seek(Slice(startKey)); it->Valid() && it->key().compare(Slice(endKey)) < 0; it->Next()) {
        std::cout << it->key().ToString() << ": " << it->value().ToString() << std::endl;
    }
    delete it;
}