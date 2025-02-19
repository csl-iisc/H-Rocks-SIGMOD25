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

## Pre-requisites
H-Rocks is built on top of pmem-rocksdb and leverages the GPU to accelerate it. 
The pre-requisities require setting up PMEM, CUDA runtime, Nvidia drivers and pmem-rocksdb. 
The following are the pre-requisites: 

### Setting up PMEM [~10 minutes]
This section explains how to setup your NVDIMM config to be run in app direct mode. This also makes sure that all the PMEM strips are interleaved to attain maximum bandwidth. 
1. Install all the dependencies to support PMEM
`chmod +x dependencies.sh`
`sudo ./dependencies.sh`
2. Run the teardown script to tear down any older PMEM configuration. 
`sudo ./pmem-setup/teardown.bashrc`
3. Run the preboot script to destroy all the existing namespaces. This script will also reboot the sytsem. 
`sudo ./pmem-setup/preboot.bashrc`
4. Run the config-setup script to configure interleaved namespace for PMEM along with app-direct mode. To run the script one has to be root. 
```
sudo su 
./pmem-setup/config-setup.bashrc
exit
```

### Setting up pmem-rocksdb [~10 minutes]
1. Git clone pmem-rocksdb
`git clone https://github.com/pmem/pmem-rocksdb`
2. Build it
`cd pmem-rocksdb`
`make ROCKSDB_ON_DCPMM=1 install-static -j`

### Setting up CUDA and Nvidia drivers 
CUDA runtime and NVIDIA drivers are necessary for H-Rocks. Follow the steps from *[NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)* for a proper installation setup.


## Reference
**[1]** RocksDB [*[Code](https://github.com/facebook/rocksdb)*]
**[2]** pmem-rocksDB [*[Code]([https://github.com/pmem/pmem-rocksdb])*]

