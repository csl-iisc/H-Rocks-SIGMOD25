# ===== Paths =====
BUILD_DIR   := ./build

HROCKS_LIB  := ../hrocksdb.a
ROCKSDB_LIB := ../../librocksdb.a

# ===== Toolchain =====
CXX  := g++
NVCC := nvcc
CUDA_ARCH ?= sm_75
BUILD_TYPE ?= release

# ===== Import RocksDB flags =====
include ../../make_config.mk
include ../../dcpmm.mk

ifndef DISABLE_JEMALLOC
	ifdef JEMALLOC
		PLATFORM_CXXFLAGS += -DROCKSDB_JEMALLOC -DJEMALLOC_NO_DEMANGLE
	endif
	EXEC_LDFLAGS := $(JEMALLOC_LIB) $(EXEC_LDFLAGS) -lpthread
	PLATFORM_CXXFLAGS += $(JEMALLOC_INCLUDE)
endif

ifneq ($(USE_RTTI), 1)
	#CXXFLAGS += -fno-rtti
endif

EXEC_LDFLAGS += $(LDFLAGS)

PLATFORM_CXXFLAGS=-std=c++14 -DHAVE_ALIGNED_NEW -DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX  -DOS_LINUX -DROCKSDB_FALLOCATE_PRESENT -DSNAPPY -DGFLAGS=1 -DZLIB -DBZIP2 -DNUMA -DTBB -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_SCHED_GETCPU_PRESENT -DROCKSDB_AUXV_GETAUXVAL_PRESENT -DHAVE_SSE42  -DHAVE_PCLMUL  -DHAVE_AVX2 -DHAVE_UINT128_EXTENSION -DROCKSDB_SUPPORT_THREAD_LOCAL

PLATFORM_CXXFLAGS2=-std=c++14 -DHAVE_ALIGNED_NEW -DROCKSDB_PLATFORM_POSIX -DROCKSDB_LIB_IO_POSIX  -DOS_LINUX -DROCKSDB_FALLOCATE_PRESENT -DSNAPPY -DGFLAGS=1 -DZLIB -DBZIP2 -DNUMA -DTBB -DROCKSDB_MALLOC_USABLE_SIZE -DROCKSDB_PTHREAD_ADAPTIVE_MUTEX -DROCKSDB_BACKTRACE -DROCKSDB_RANGESYNC_PRESENT -DROCKSDB_SCHED_GETCPU_PRESENT -DROCKSDB_AUXV_GETAUXVAL_PRESENT -DHAVE_SSE42  -DHAVE_PCLMUL  -DHAVE_AVX2 -DHAVE_UINT128_EXTENSION -DROCKSDB_SUPPORT_THREAD_LOCAL


# ===== Includes (match tree headers) =====
CPPFLAGS += -I../src -I../libgpm/include -I../../include -I../bin

# ===== Host flags (g++) =====
HOST_PLATFORM := $(filter-out -std=%,$(PLATFORM_CXXFLAGS))
CXXFLAGS += -std=c++11 -DON_DCPMM 
CXXFLAGS += $(EXTRA_PLATFORM_DEFS)
CPPFLAGS += $(CXXFLAGS)

# ===== NVCC =====
NVCCFLAGS := -lpmem -rdc=true -lpthread -lnvidia-ml -arch=sm_75 -lpci -Xcompiler -fopenmp -lrt -lm -O3
# NVCC_XCOMPILER := $(foreach f,$(HOST_PLATFORM),-Xcompiler $(f))
NVCC_OMP := -Xcompiler -fopenmp

ifeq ($(BUILD_TYPE),debug)
  CXXFLAGS  += -g
  NVCCFLAGS += -g -G
else
  CXXFLAGS  += -O3
  NVCCFLAGS += -O3
endif

# ===== Link flags/libs =====
LDFLAGS += $(PLATFORM_LDFLAGS) -L/usr/local/lib -L/usr/local/lib64
LDLIBS  += $(EXEC_LDFLAGS) \
           -lpthread -lpmem -lpmemobj -lnvidia-ml -lrt -lm -lgomp -ldl 

.PHONY: all clean hrocksdb librocksdb
BINS := \
  $(BUILD_DIR)/simple_test_cc \
  $(BUILD_DIR)/simple_test_cu \
  $(BUILD_DIR)/test_puts \
  $(BUILD_DIR)/test_put_get \
  $(BUILD_DIR)/test_ycsbA \
  $(BUILD_DIR)/test_ycsbB \
  $(BUILD_DIR)/test_ycsbC

all: $(BINS)

hrocksdb:
	$(MAKE) -C .. all

librocksdb:
	$(MAKE) -C ../.. static_lib -j

$(BUILD_DIR):
	mkdir -p build

# ---- Host binary ----
$(BUILD_DIR)/simple_test_cc: simple_test.cc | $(BUILD_DIR) librocksdb
	$(CXX) -std=c++11 -fno-rtti $(CPPFLAGS) $< -o $@ \
	  $(ROCKSDB_LIB) \
	  -L/usr/local/lib/ -L/usr/local/lib64/ \
	  $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) \
	  -lpthread -lrt -lsnappy -lgflags -lz -lbz2 -llz4 -lzstd -lnuma -ltbb -ldl -lpmem -lpmemobj


# ---- CUDA binaries ----
$(BUILD_DIR)/simple_test_cu: simple_test.cu | $(BUILD_DIR) hrocksdb librocksdb
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart 
	  

$(BUILD_DIR)/test_puts: test_puts.cu | $(BUILD_DIR) hrocksdb librocksdb
	$(NVCC) $(NVCCFLAGS) $(NVCC_XCOMPILER) $(NVCC_OMP) $(CPPFLAGS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/test_put_get: test_put_get.cu | $(BUILD_DIR) hrocksdb librocksdb
	$(NVCC) $(NVCCFLAGS) $(NVCC_XCOMPILER) $(NVCC_OMP) $(CPPFLAGS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/test_ycsbA: test_ycsbA.cu | $(BUILD_DIR) hrocksdb librocksdb
	$(NVCC) $(NVCCFLAGS) $(NVCC_XCOMPILER) $(NVCC_OMP) $(CPPFLAGS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/test_ycsbB: test_ycsbB.cu | $(BUILD_DIR) hrocksdb librocksdb
	$(NVCC) $(NVCCFLAGS) $(NVCC_XCOMPILER) $(NVCC_OMP) $(CPPFLAGS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

$(BUILD_DIR)/test_ycsbC: test_ycsbC.cu | $(BUILD_DIR) hrocksdb librocksdb
	$(NVCC) $(NVCCFLAGS) $(NVCC_XCOMPILER) $(NVCC_OMP) $(CPPFLAGS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf $(BUILD_DIR)
