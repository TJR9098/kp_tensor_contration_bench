CXX ?= g++
TBLIS_PREFIX ?= /root/benchmark/exo_tensor_bench_openeuler2203_arm64_20261008_v2/install/tblis-2.0-armsve
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS += -DTBLIS_ENABLE_CPLUSPLUS=1 -I$(TBLIS_PREFIX)/include
LDLIBS += -L$(TBLIS_PREFIX)/lib -Wl,-rpath,$(TBLIS_PREFIX)/lib -ltblis -latomic -lpthread

.PHONY: all clean

all: bin/tblis_contraction_runner

bin/tblis_contraction_runner: tblis_runner.cpp
	mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDLIBS) -o $@

clean:
	$(RM) bin/tblis_contraction_runner
