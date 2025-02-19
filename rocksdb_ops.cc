#include "rocksdb_ops.h" 
#include "debugger.h"
#include "db_timer.h"

RocksDBOperations::RocksDBOperations(rocksdb::DB* db, Debugger debug, DbTimer* timer): db(db), debug(debug), timer(timer) {}

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