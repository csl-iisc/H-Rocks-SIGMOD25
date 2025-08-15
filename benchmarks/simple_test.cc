#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/options.h>

using namespace std;
using namespace ROCKSDB_NAMESPACE;

int main() {
  DB* db;
  Options options;
  // Optimize RocksDB. This is the easiest way to get RocksDB to perform well
  options.IncreaseParallelism();
  options.OptimizeLevelStyleCompaction();
  // create the DB if it's not already present
  options.create_if_missing = true;
  std::cout << "Opening DB at /pmem/rocksdb_simple_example" << std::endl;
  std::string db_path = "/pmem/rocksdb_simple_example";

  // // open DB
  Status s = DB::Open(options, db_path, &db);
  // cout << "DB opened." << endl;
  return 0;
}