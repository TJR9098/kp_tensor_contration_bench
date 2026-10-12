CXX ?= g++
TBLIS_PREFIX ?= /root/benchmark/exo_tensor_bench_openeuler2203_arm64_20261008_v2/install/tblis-2.0-armsve
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS += -DTBLIS_ENABLE_CPLUSPLUS=1 -I$(TBLIS_PREFIX)/include
LDLIBS += -L$(TBLIS_PREFIX)/lib -Wl,-rpath,$(TBLIS_PREFIX)/lib -ltblis -latomic -lpthread

.PHONY: all clean cases tblis-cases tblis-batch-cases

all: bin/tblis_contraction_runner bin/tblis_layout_runner

cases: cases.tsv

tblis-cases: tblis_cases.tsv

tblis-batch-cases: tblis_batch_cases.tsv

bin/tblis_contraction_runner: tblis_runner.cpp
	mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDLIBS) -o $@

bin/tblis_layout_runner: tblis_layout_runner.cpp
	mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDLIBS) -lcrypto -o $@

cases.tsv: generate_cases.py
	python3 $< $@

tblis_cases.tsv: generate_tblis_cases.py
	python3 $< $@

tblis_batch_cases.tsv: generate_tblis_batch_cases.py
	python3 $< $@

clean:
	$(RM) bin/tblis_contraction_runner bin/tblis_layout_runner
