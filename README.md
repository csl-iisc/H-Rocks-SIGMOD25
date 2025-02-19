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

## H-Rocks setup and source code 

### Pre-requisites
H-Rocks is built on top of pmem-rocksdb [2] and leverages the GPU to accelerate it. 
The pre-requisities require setting up PMEM, CUDA runtime, Nvidia drivers and pmem-rocksdb. 
The following are the pre-requisites: 

### Setting up PMEM [~10 minutes]
This section explains how to setup your NVDIMM config to be run in app direct mode. This also ensures all the PMEM strips are interleaved to attain maximum bandwidth. 
1. Install all the dependencies to support PMEM
```bash
chmod +x dependencies.sh
sudo ./dependencies.sh
```
3. Run the teardown script to tear down any older PMEM configuration. 
```bash
sudo ./pmem-setup/teardown.bashrc
```
5. Run the preboot script to destroy all the existing namespaces. This script will also reboot the sytsem. 
```bash
sudo ./pmem-setup/preboot.bashrc
```
6. Run the config-setup script to configure interleaved namespace for PMEM along with app-direct mode. To run the script one has to be root. 
```bash
sudo su 
./pmem-setup/config-setup.bashrc
exit
```

### Setting up pmem-rocksdb [~10 minutes]
1. Git clone pmem-rocksdb.
```bash 
git clone https://github.com/pmem/pmem-rocksdb
```
2. Create the build folder. 
```bash
cd pmem-rocksdb && mkdir build
```
3. Compile pmem-rocksdb.
```bash
make ROCKSDB_ON_DCPMM=1 install-static -j
```

### Setting up CUDA and Nvidia drivers [~15 minutes]
CUDA runtime and NVIDIA drivers are necessary for H-Rocks. Follow the steps from *[NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)* for a proper installation setup.

### Compiling H-Rocks
1. Git clone H-Rocks within the pmem-rocksdb folder.
```bash
git clone https://github.com/csl-iisc/H-Rocks-SIGMOD25
```
2. Change the directory to H-Rocks-SIGMOD25 and compile it. 
```bash
cd H-Rocks-SIGMOD25/ && make hrocksdb
```
The compiled library can be found in H-Rocks-SIGMOD25 as *hrocksdb.a*. 
H-Rocks can also be compiled in the DEBUG mode using the debug flag. This allows both GCC and CUDA debugging and logging information from H-Rocks. 
```bash
make hrocksdb -DENABLE_DEBUG
```

### Running H-Rocks
One can compile and run existing test cases provided in the benchmark folder. 
Follow the steps below: 
```bash
make bin/test_puts
./bin/test_puts -n <num_keys> -k <key_size> -v <value_size>
make bin/test_gets
./bin/test_gets -p <num_prefill_keys> -n <num_keys> -k <key_size> -v <value_size>
```
Other tests can be run similarly. 

### H-Rocks Source Code

The H-Rocks source code is organized within the `H-Rocks-SIGMOD25/src` directory. Key files including `hrocksdb.h` and `hrocksdb.cu` define the main APIs of H-Rocks, which are designed to enhance the functionality of the existing RocksDB interfaces by leveraging both CPU and GPU resources.

#### H-Rocks Main API Overview

- **Constructor and destructor**:
  - `HRocksDB(Config config);` - initializes a new instance of H-Rocks with the specified configuration.
  - `~HRocksDB();` - destroys an instance of H-Rocks, freeing up resources.

- **Database operations**:
  - `void Close();` - closes the H-Rocks database.
  - `void HOpen(std::string fileLocation);` - opens a database at the specified location.
  - `void Delete(std::string fileLocation);` - deletes the database at the specified location.

- **Key-Value store operations**:
  - `void Put(const std::string& key, const std::string& value);` - inserts or updates a key-value pair.
  - `void Delete(const std::string& key);` - removes a key-value pair by key.
  - `void Range(const std::string& startKey, const std::string& endKey);` - retrieves a range of key-value pairs between the specified start and end keys.
  - `void Merge(const std::string& key);` - merges a key with its existing value using a predefined merge function.
  - `void Get(const std::string& key);` - retrieves the value associated with a specified key.

#### Configuration Methods

H-Rocks can be fine-tuned with several configuration methods, enabling optimal performance tailored to specific hardware and workload requirements:

- `setMemtableSize(uint64_t size);` - sets the size of the memtable.
- `setNumMemtables(int num);` - sets the number of memtables to maintain concurrently.
- `setBatchSize(uint64_t size);` - sets the batch size for all KVS operations.

For more details on each API and configuration settings, refer to the comments within the source files located at [src/hrocksdb.h](src/hrocksdb.h) and [src/hrocksdb.cu](src/hrocksdb.cu).

## Setting up docker container
To install docker on an Ubuntu machine
```bash
sudo apt install docker.io
```

To run experiments within the container, build the container as:

```bash
docker build . -t sa:v1
```

The docker container requires access to NVIDIA GPUs and NVIDIA driver. This is enabled by installing NVIDIA container toolkit. Follow the steps from *[NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)* to set it up. We recommend installing via APT.

Then launch the docker container in an interactive as:
```bash
docker container run -it --runtime=nvidia --gpus all sa:v1 bash
```



## Reference
**[1]** RocksDB [*[Code](https://github.com/facebook/rocksdb)*]

**[2]** pmem-rocksDB [*[Code](https://github.com/pmem/pmem-rocksdb)*]

