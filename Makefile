include ../../make_config.mk

ifndef DISABLE_JEMALLOC
	ifdef JEMALLOC
		PLATFORM_CXXFLAGS += -DROCKSDB_JEMALLOC -DJEMALLOC_NO_DEMANGLE
	endif
	EXEC_LDFLAGS := $(JEMALLOC_LIB) $(EXEC_LDFLAGS) -lpthread
	PLATFORM_CXXFLAGS += $(JEMALLOC_INCLUDE)
endif

ifneq ($(USE_RTTI), 1)
	# CXXFLAGS += -fno-rtti
endif

include ../../dcpmm.mk
EXEC_LDFLAGS += $(LDFLAGS)

PLATFORM_CXXFLAGS=-std=c++14 -DHAVE_ALIGNED_NEW -DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX -DOS_LINUX -DROCKSDB_FALLOCATE_PRESENT -DSNAPPY -DGFLAGS=1 -DZLIB -DBZIP2 -DNUMA -DTBB -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_SCHED_GETCPU_PRESENT -DROCKSDB_AUXV_GETAUXVAL_PRESENT -DHAVE_SSE42  -DHAVE_PCLMUL  -DHAVE_AVX2 -DHAVE_UINT128_EXTENSION -DROCKSDB_SUPPORT_THREAD_LOCAL

PLATFORM_CXXFLAGS2=-std=c++14 -DHAVE_ALIGNED_NEW -DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX -DOS_LINUX -DROCKSDB_FALLOCATE_PRESENT -DSNAPPY -DGFLAGS=1 -DZLIB -DBZIP2 -DNUMA -DTBB -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_SCHED_GETCPU_PRESENT -DROCKSDB_AUXV_GETAUXVAL_PRESENT -DHAVE_SSE42  -DHAVE_PCLMUL  -DHAVE_AVX2 -DHAVE_UINT128_EXTENSION -DROCKSDB_SUPPORT_THREAD_LOCAL

CXX = g++
#CXXFLAGS = -lpthread #-O3
BIN_DIR = bin
TEST_DIR = benchmarks
NVCC = nvcc
INCLUDE_FILE = -I. -I./bin -I../libgpm/include -I/usr/local/cuda/include -I../../include -I/usr/local/include
NVCCFLAGS = -lpmem -rdc=true -lpthread -lnvidia-ml -arch=sm_75 -lpci -Xcompiler -fopenmp -lrt -lm 
LIB_NAME = hrocksdb.a


# ENABLE_DEBUG := 1

ifdef ENABLE_DEBUG 
	NVCCFLAGS += -g -G #-DDEBUG_MODE 
	CXXFFLAGS += -g -fsanitize=address
else 
	NVCCFLAGS += -O3
	CXXFLAGS += -O3
endif 

.PHONY: clean librocksdb

librocksdb:
	cd ../.. && make static_lib

# Define the list of .o files
OBJS := \
	$(BIN_DIR)/sub_batch.o \
	$(BIN_DIR)/batch.o \
	$(BIN_DIR)/config.o \
	$(BIN_DIR)/hrocksdb.o \
	$(BIN_DIR)/debugger.o \
	$(BIN_DIR)/gmemtable.o \
	$(BIN_DIR)/helper.o \
	$(BIN_DIR)/sort_larger.o \
	$(BIN_DIR)/sort_smaller.o \
	$(BIN_DIR)/sort_smaller_updates.o \
	$(BIN_DIR)/sort_larger_updates.o \
	$(BIN_DIR)/log.o \
	$(BIN_DIR)/gpu_puts.o \
	$(BIN_DIR)/gpu_gets.o \
	$(BIN_DIR)/search.o \
	$(BIN_DIR)/block_cache.o \
	$(BIN_DIR)/sst_writer.o \
	$(BIN_DIR)/cpu_gets.o \
	$(BIN_DIR)/write_sub_batch.o \
	$(BIN_DIR)/gpu_updates.o \
	$(BIN_DIR)/update_sub_batch.o \
	$(BIN_DIR)/timer.o \
	$(BIN_DIR)/rocksdb_ops.o 

# Define hrocksdb to have all the .o files
hrocksdb: $(OBJS)
	ar rcs $(LIB_NAME) $(OBJS)

$(BIN_DIR)/config.o: config.cc config.h
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@

$(BIN_DIR)/timer.o: db_timer.cc db_timer.h
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@

$(BIN_DIR)/sub_batch.o: sub_batch.cu sub_batch.h
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@

$(BIN_DIR)/write_sub_batch.o: write_sub_batch.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@

$(BIN_DIR)/update_sub_batch.o: update_sub_batch.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@

$(BIN_DIR)/batch.o: batch.cu batch.h
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@

$(BIN_DIR)/block_cache.o: block_cache.cu block_cache.h
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@

$(BIN_DIR)/debugger.o: debugger.cc debugger.h
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@

$(BIN_DIR)/log.o: log.cu log.cuh
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@ 

$(BIN_DIR)/gmemtable.o: gmemtable.cu gmemtable.h
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@ 

$(BIN_DIR)/helper.o: helper.cu helper.cuh
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@ 

$(BIN_DIR)/sort_larger.o: sort_larger.cu 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@ 

$(BIN_DIR)/sort_larger_updates.o: sort_larger_updates.cu 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@ 


$(BIN_DIR)/search.o: search.cu 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@ 

$(BIN_DIR)/sst_writer.o: sst_writer.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $< -o $@  -I../../include -std=c++11  $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

$(BIN_DIR)/rocksdb_ops.o: rocksdb_ops.cc
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $< -o $@  -I../../include -std=c++11  $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

$(BIN_DIR)/cpu_gets.o: cpu_gets.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $< -o $@  -I../../include $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

$(BIN_DIR)/sort_smaller.o: sort_smaller.cu 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@ 

$(BIN_DIR)/sort_smaller_updates.o: sort_smaller_updates.cu 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@ 

$(BIN_DIR)/gpu_puts.o: gpu_puts.cu gpu_puts.cuh
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@ 

$(BIN_DIR)/gpu_updates.o: gpu_updates.cu gpu_updates.cuh
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@

$(BIN_DIR)/gpu_gets.o: gpu_gets.cu gpu_gets.cuh
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@ 

$(BIN_DIR)/hrocksdb.o: hrocksdb.cu hrocksdb.h 
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(PLATFORM_LDFLAGS) -I../../include $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -c $< -o $@  $(INCLUDE_FILE) 
	# $(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) -c $< -o $@ 

$(BIN_DIR)/simple_test1: benchmarks/simple_test.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) $(LIB_NAME) ../../librocksdb.a $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) -std=c++14 -L/usr/local/cuda/lib64 -lcuda -lcudart $< -o $@

$(BIN_DIR)/simple_test2: benchmarks/simple_test.cc
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@  ../../librocksdb.a  -I../../include -O2 -std=c++11 $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS)

$(BIN_DIR)/test_block_cache: benchmarks/test_block_cache.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDE_FILE) ../../librocksdb.a $(LIB_NAME) $(PLATFORM_CXXFLAGS) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS)  -L/usr/local/cuda/lib64 -lcuda -lcudart -o $@ $<

$(BIN_DIR)/test_puts: benchmarks/test_puts.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) ../../librocksdb.a $(LIB_NAME) $(PLATFORM_CXXFLAGS) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS)  -L/usr/local/cuda/lib64 -lcuda -lcudart -o $@ $<

$(BIN_DIR)/test_put_get: benchmarks/test_put_get.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) ../../librocksdb.a $(LIB_NAME) $(PLATFORM_CXXFLAGS) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS)  -L/usr/local/cuda/lib64 -lcuda -lcudart -o $@ $<

$(BIN_DIR)/test_ycsbA: benchmarks/test_ycsbA.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) ../../librocksdb.a $(LIB_NAME) $(PLATFORM_CXXFLAGS) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS)  -L/usr/local/cuda/lib64 -lcuda -lcudart -o $@ $<

$(BIN_DIR)/test_ycsbB: benchmarks/test_ycsbB.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) ../../librocksdb.a $(LIB_NAME) $(PLATFORM_CXXFLAGS) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS)  -L/usr/local/cuda/lib64 -lcuda -lcudart -o $@ $<

$(BIN_DIR)/test_ycsbC: benchmarks/test_ycsbC.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) ../../librocksdb.a $(LIB_NAME) $(PLATFORM_CXXFLAGS) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS)  -L/usr/local/cuda/lib64 -lcuda -lcudart -o $@ $<

$(BIN_DIR)/recover: recover.cu
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FILE) ../../librocksdb.a $(LIB_NAME) $(PLATFORM_CXXFLAGS) $(PLATFORM_LDFLAGS) $(EXEC_LDFLAGS)  -L/usr/local/cuda/lib64 -lcuda -lcudart -o $@ $<

clean: 
	rm -rf $(BIN_DIR) hrocksdb.a