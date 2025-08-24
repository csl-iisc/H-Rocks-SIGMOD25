include ../../make_config.mk
include ../../dcpmm.mk

ROCKSDB_LIB := ../../librocksdb.a
HROCKS_LIB  := ../hrocksdb.a         # only needed for your CUDA tests; not used below
BIN_DIR     := bin

CXX := g++
NVCC := nvcc

# Compile-time flags only
CXXFLAGS += -O3 -std=c++11 -DON_DCPMM
NVCCFLAGS = -lpmem -rdc=true -lpthread -lnvidia-ml -arch=sm_75 -lpci -Xcompiler -fopenmp -lrt -lm -O3 -std=c++11 -DON_DCPMM

ifneq ($(USE_RTTI), 1)
  CXXFLAGS += -fno-rtti
#   NVCCFLAGS += -Xcompiler -fno-rtti
endif

# Headers from this RocksDB tree / project
INCLUDE_PATHS += -I../../include -I../src -I../bin -I/usr/local/cuda/include -I../libgpm/include

ifdef DEBUG 
	NVCCFLAGS += -g -G 
	CXXFFLAGS += -g -fsanitize=address
endif 

# Link flags/libs (don’t mix compile flags here)
LDFLAGS += $(PLATFORM_LDFLAGS)
LDLIBS  += $(EXEC_LDFLAGS) \
           -lpthread -lrt -lsnappy -lgflags -lz -lbz2 -llz4 -lzstd -lnuma -ltbb -ldl \
           -lpmem -lpmemobj -lgomp -lnvidia-ml -lpci

.PHONY: all clean librocksdb begin

all: librocksdb begin \
	bin/simple_test_cc \
	bin/simple_test_cu \
	bin/test_puts \
	bin/test_put_get \
	bin/test_ycsbA \
	bin/test_ycsbB \
	bin/test_ycsbC

librocksdb:
	$(MAKE) -C ../../ static_lib

begin:
	mkdir -p $(BIN_DIR)

# ---- C++ example (uses only librocksdb.a) ----
bin/simple_test_cc: simple_test.cc | begin librocksdb
	$(CXX) $(INCLUDE_PATHS) $(CXXFLAGS) $< -o $@ \
	  $(ROCKSDB_LIB) $(LDFLAGS) $(LDLIBS)

# ---- CUDA example (links with hrocksdb.a) ----
bin/simple_test_cu: simple_test.cu | begin librocksdb
	$(NVCC) $(INCLUDE_PATHS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

bin/test_puts: test_puts.cu | begin librocksdb
	$(NVCC) $(INCLUDE_PATHS) $(NVCCFLAGS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

bin/test_put_get: test_put_get.cu | begin librocksdb
	$(NVCC) $(INCLUDE_PATHS) $(NVCCFLAGS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

bin/test_ycsbA: test_ycsbA.cu | begin librocksdb
	$(NVCC) $(INCLUDE_PATHS) $(NVCCFLAGS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

bin/test_ycsbB: test_ycsbB.cu | begin librocksdb
	$(NVCC) $(INCLUDE_PATHS) $(NVCCFLAGS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

bin/test_ycsbC: test_ycsbC.cu | begin librocksdb
	$(NVCC) $(INCLUDE_PATHS) $(NVCCFLAGS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

bin/test_req_rate: test_req_rate.cu | begin librocksdb
	$(NVCC) $(INCLUDE_PATHS) $(NVCCFLAGS) $< -o $@ \
	  $(HROCKS_LIB) $(ROCKSDB_LIB) \
	  -L/usr/local/cuda/lib64 -lcuda -lcudart \
	  $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf $(BIN_DIR)
