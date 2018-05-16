GTEST_REPEAT := 50

COMMON_BZL_FLAGS := --test_output=all --cache_test_results=no

GTEST_FLAGS := --action_env="GTEST_SHUFFLE=1" --action_env="GTEST_BREAK_ON_FAILURE=1"

REP_BZL_FLAG := --action_env="GTEST_REPEAT=$(GTEST_REPEAT)"

MEMCHECK_BZL_FLAG := --run_under="valgrind"

COVERAGE_BZL_FLAG := --instrumentation_filter= --spawn_strategy=standalone

BUILD := bazel build

RUN := bazel run

TEST := bazel test $(COMMON_BZL_FLAGS)

GTEST := $(TEST) $(GTEST_FLAGS)

COVER := bazel coverage $(COMMON_BZL_FLAGS) $(COVERAGE_BZL_FLAG)

SERIALS := serial_cc serial_py

GRAPHMGRS := graphmgr_cc graphmgr_py

all: proto test_py proto_build unit_test test_regress

travis_test: build test_py proto_build memcheck coverage test_regress

# build protobuf files
build: tenncor_build proto

tenncor_build:
	$(BUILD) //:tenncor

proto: monitor $(SERIALS) $(GRAPHMGRS)

monitor:
	$(BUILD) //proto:tenncor_monitor_grpc

$(SERIALS):
	$(BUILD) //proto:tenncor_$@_proto

$(GRAPHMGRS):
	$(BUILD) //tests/graphmgr:$@_grpc

# python data build and test
proto_build:
	$(RUN) //tests/py:protogen -- $(shell pwd)/tests/unit/samples

test_py:
	$(TEST) //tests/py/test:test

test: test_clay test_mold test_kiln 

test_clay:
	$(GTEST) $(REP_BZL_FLAG) //clay:test

test_mold:
	$(GTEST) $(REP_BZL_FLAG) //mold:test

test_kiln:
	$(GTEST) $(REP_BZL_FLAG) //kiln:test

# unit test
unit_test: test_tensor test_graph test_operate test_serialize

test_tensor:
	$(GTEST) $(REP_BZL_FLAG) //tests/unit:test_tensor

test_graph:
	$(GTEST) $(REP_BZL_FLAG) //tests/unit:test_graph

test_operate:
	$(GTEST) $(REP_BZL_FLAG) //tests/unit:test_operate

test_serialize:
	$(GTEST) $(REP_BZL_FLAG) //tests/unit:test_serialize

# conducts coverage on unit tests
coverage: cover_tensor cover_graph cover_operate cover_serialize

cover_tensor:
	$(COVER) $(REP_BZL_FLAG) //tests/unit:test_tensor

cover_graph:
	$(COVER) $(REP_BZL_FLAG) //tests/unit:test_graph

cover_operate:
	$(COVER) $(REP_BZL_FLAG) //tests/unit:test_operate

cover_serialize: # serialize is already expensive. don't repeat
	$(COVER) //tests/unit:test_serialize

# runs unit tests with valgrind memory leak, require valgrind to be installed
memcheck: memcheck_tensor memcheck_graph memcheck_operate memcheck_serialize

memcheck_tensor:
	$(GTEST) $(MEMCHECK_BZL_FLAG) --action_env="GTEST_REPEAT=5" //tests/unit:test_tensor

memcheck_graph:
	$(GTEST) $(MEMCHECK_BZL_FLAG) --action_env="GTEST_REPEAT=5" //tests/unit:test_graph

memcheck_operate:
	$(GTEST) $(MEMCHECK_BZL_FLAG) --action_env="GTEST_REPEAT=5" //tests/unit:test_operate

memcheck_serialize: # serialize is already expensive. don't repeat
	$(GTEST) $(MEMCHECK_BZL_FLAG) //tests/unit:test_serialize

# regression test
test_regress:
	$(GTEST) //tests/regress:test

# todo: deprecate
acceptdata: cleandata
	python tests/regress/tf_generate/tf_generate.py

# clean C++ format with astyle, requires astyle to be installed
fmt:
	astyle --project --recursive --suffix=none *.hpp,*.ipp,*.cpp

# remove all test data
clean_data:
	rm -f tests/regress/samples/*
	rm -f tests/unit/samples/*

clean: clean_data
	bazel clean
