# ================== Config ==================
include ../make_config.mk       # from the RocksDB tree (same one used to build librocksdb.a)
include ../dcpmm.mk

SRCDIR := src
BIN_DIR := bin
LIB_NAME := hrocksdb.a

CXX  := g++
NVCC := nvcc

CUDA_ARCH ?= sm_75
BUILD_TYPE ?= release

# jemalloc (same logic as your snippet)
ifndef DISABLE_JEMALLOC
  ifdef JEMALLOC
    EXTRA_PLATFORM_DEFS += -DROCKSDB_JEMALLOC -DJEMALLOC_NO_DEMANGLE
  endif
  EXEC_LDFLAGS := $(JEMALLOC_LIB) $(EXEC_LDFLAGS) -lpthread
  CPPFLAGS     += $(JEMALLOC_INCLUDE) 
endif


# ===== Host flags (one -std only; merge RocksDB flags + your extra defs) =====
HOST_PLATFORM := $(filter-out -std=%,$(PLATFORM_CXXFLAGS)) $(EXTRA_PLATFORM_DEFS)
CXXFLAGS += -std=c++11 $(HOST_PLATFORM) -DON_DCPMM -fno-rtti -faligned-new -fno-builtin-memcmp -march=native

# ---- Includes: use the SAME headers as the lib ----
CPPFLAGS := -I. -I$(SRCDIR) -I$(BIN_DIR) -I../include -I/usr/local/cuda/include -I./libgpm/include

# NVCC: pass host flags via -Xcompiler; do not pass PLATFORM_CXXFLAGS directly
NVCCFLAGS := -std=c++14 -arch=$(CUDA_ARCH) -rdc=true
NVCC_OMP := -Xcompiler -fopenmp

ifeq ($(BUILD_TYPE),debug)
  CXXFLAGS  += -g
  NVCCFLAGS += -g -G
else
  CXXFLAGS  += -O3
  NVCCFLAGS += -O3
endif

# Link flags/libs (not really used for a static .a, but harmless)
LDFLAGS += $(PLATFORM_LDFLAGS)
LDLIBS  += $(EXEC_LDFLAGS) -lpthread -lpmem -lnvidia-ml -lrt -lm -lgomp

ROCKSDB_LIB := ../librocksdb.a

# ================== Targets ==================
.PHONY: all clean librocksdb
all: hrocksdb

librocksdb:
	$(MAKE) -C ../ static_lib

# Object list
OBJS := \
	$(BIN_DIR)/sub_batch.o \
	$(BIN_DIR)/batch.o \
	$(BIN_DIR)/config.o \
	$(BIN_DIR)/debugger.o \
	$(BIN_DIR)/gmemtable.o \
	$(BIN_DIR)/helper.o \
	$(BIN_DIR)/sort_larger.o \
	$(BIN_DIR)/sort_smaller.o \
	$(BIN_DIR)/sort_smaller_updates.o \
	$(BIN_DIR)/sort_larger_updates.o \
	$(BIN_DIR)/log.o \
	$(BIN_DIR)/hrocksdb.o \
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
	$(BIN_DIR)/db_timer.o \
	$(BIN_DIR)/rocksdb_ops.o

hrocksdb: $(OBJS)
	ar rcs $(LIB_NAME) $(OBJS)

# ================== Pattern Rules ==================
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# CUDA sources -- removed cpp flags since nvcc handles them
$(BIN_DIR)/%.o: $(SRCDIR)/%.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCC_XCOMPILER) $(NVCC_OMP) -c $< -o $@ 

# C++ sources
$(BIN_DIR)/%.o: $(SRCDIR)/%.cc | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# ================== Special-case (still pass host flags correctly) ==================

$(BIN_DIR)/sst_writer.o: $(SRCDIR)/sst_writer.cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_XCOMPILER) $(NVCC_OMP) $(CPPFLAGS) $(CXXFFLAGS) -c $< -o $@

$(BIN_DIR)/rocksdb_ops.o: $(SRCDIR)/rocksdb_ops.cc
	$(CXX) $(CXXFLAGS) $(PLATFORM_LDFLAGS) $(PLATFORM_CXXFLAGS) $(EXEC_LDFLAGS) $(CPPFLAGS) $(CXXFFLAGS) -std=c++11 -c $< -o $@

clean:
	rm -rf $(BIN_DIR) $(LIB_NAME)

# ================== Tests that live in src/ ==================

# Link flags/libs (match your working setup)
LDFLAGS  += $(PLATFORM_LDFLAGS) -L/usr/local/lib -L/usr/local/lib64
LDLIBS   += $(EXEC_LDFLAGS) \
            -lpthread -lpmem -lpmemobj -lnvidia-ml -lpci -lrt -lm -lgomp -ldl \
            -lsnappy -lzstd -llz4 -lbz2 -lz

ROCKSDB_LIB := ../librocksdb.a  # same tree you built

# Build all src tests with: make test-src
.PHONY: test-src
test-src: \
  $(BIN_DIR)/src_simple_test_cc \
  $(BIN_DIR)/src_simple_test_cu \
  $(BIN_DIR)/src_test_puts \
  $(BIN_DIR)/src_test_put_get 

# ---- Host-only test (simple_test.cc) ----
$(BIN_DIR)/src_simple_test_cc: $(SRCDIR)/simple_test.cc $(LIB_NAME) $(ROCKSDB_LIB) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ \
	  $(LIB_NAME) $(ROCKSDB_LIB) \
	  $(LDFLAGS) $(LDLIBS)

# ---- CUDA tests (link with nvcc) ----
$(BIN_DIR)/src_simple_test_cu: $(SRCDIR)/simple_test.cu $(LIB_NAME) $(ROCKSDB_LIB) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCC_XCOMPILER) $(NVCC_OMP) $(CPPFLAGS) $< -o $@ \
	  $(LIB_NAME) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/src_test_puts: $(SRCDIR)/test_puts.cu $(LIB_NAME) $(ROCKSDB_LIB) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCC_XCOMPILER) $(NVCC_OMP) $(CPPFLAGS) $< -o $@ \
	  $(LIB_NAME) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/src_test_put_get: $(SRCDIR)/test_put_get.cu $(LIB_NAME) $(ROCKSDB_LIB) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCC_XCOMPILER) $(NVCC_OMP) $(CPPFLAGS) $< -o $@ \
	  $(LIB_NAME) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/src_test_block_cache: $(SRCDIR)/test_block_cache.cu $(LIB_NAME) $(ROCKSDB_LIB) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCC_XCOMPILER) $(NVCC_OMP) $(CPPFLAGS) $< -o $@ \
	  $(LIB_NAME) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)
