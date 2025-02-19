# H-Rocks: CPU-GPU accelerated Heterogeneous RocksDB on Persistent Memory

H-Rocks extends the popular key-value store RocksDB by Meta [1] to leverage both GPU and CPU. 
H-Rocks significantly improves the throughput of RocksDB.
This repository provides the source code for H-RocksDB, designed to accelerate a wide range of RocksDB operations by selectively offloading them to the GPU. 
This README provides a peek into the key-value store and a high-level view of source code organization.

For full details, refer to our paper:
<pre>
<b>H-Rocks: CPU-GPU accelerated Heterogeneous RocksDB on Persistent Memory</b>
Shweta Pandey and Arkaprava Basu
<i>Proceedings of the ACM on Management of Data, Volume 3, Issue 1 (SIGMOD), 2025</i>
DOI: https://doi.org/10.1145/3709694
</pre>

## Hardware and software requirements
H-Rocks is built on top of pmem-rocksdb and shares its requirements, listed below:
* SM compute capability: >= 7.5 && <= 8.6
* Host CPU: x86\_64, ppc64le, aarch64
* OS: Linux v 5.4.0-169-generic
* GCC version : >= 5.3.0 for x86\_64;
* CUDA version: >= 8.0 && <= 12.1
* CUDA driver version: >= 530.xx

## Reference
**[1]** RocksDB [*[Code](https://github.com/facebook/rocksdb)*]
**[2]** pmem-rocksDB [*[Code]([https://github.com/pmem/pmem-rocksdb])*]

