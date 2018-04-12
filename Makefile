GTEST_REPEAT := 50

COMMON_BZL_FLAGS := --test_output=all --cache_test_results=no

GTEST_FLAGS := --action_env="GTEST_SHUFFLE=1" --action_env="GTEST_BREAK_ON_FAILURE=1"

REP_BZL_FLAG := --action_env="GTEST_REPEAT=$(GTEST_REPEAT)"

MEMCHECK_BZL_FLAG := --run_under="valgrind"

COVERAGE_BZL_FLAG := --instrumentation_filter= --spawn_strategy=standalone

RUN := bazel run

TEST := bazel test $(COMMON_BZL_FLAGS)

GTEST := $(TEST) $(GTEST_FLAGS)

COVER := bazel coverage $(COMMON_BZL_FLAGS) $(COVERAGE_BZL_FLAG)

all: test_py proto_build unit_test test_regress


# python data build and test
proto_build:
	$(RUN) //tests/py:protogen -- $(shell pwd)/tests/unit/samples

test_py:
	$(TEST) //tests/py/test:test


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
	$(COVER) //tests/unit:test_tensor

cover_graph:
	$(COVER) //tests/unit:test_graph

cover_operate:
	$(COVER) //tests/unit:test_operate

cover_serialize:
	$(COVER) //tests/unit:test_serialize


# runs unit tests with valgrind memory leak, require valgrind to be installed
memcheck: memcheck_tensor memcheck_graph memcheck_operate memcheck_serialize

memcheck_tensor:
	$(GTEST) $(MEMCHECK_BZL_FLAG) //tests/unit:test_tensor

memcheck_graph:
	$(GTEST) $(MEMCHECK_BZL_FLAG) //tests/unit:test_graph

memcheck_operate:
	$(GTEST) $(MEMCHECK_BZL_FLAG) //tests/unit:test_operate

memcheck_serialize:
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
