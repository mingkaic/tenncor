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

all: pytest unittest accepttest



coverage: tensorcover graphcover operatecover serializecover

tensorcover:
	$(COVER) //tests/unit:test_tensor

graphcover:
	$(COVER) //tests/unit:test_graph

operatecover:
	$(COVER) //tests/unit:test_operate

serializecover:
	$(COVER) //tests/unit:test_serialize



memcheck: tensortest graphtest operatetest serializememcheck

tensormemcheck:
	$(GTEST) $(MEMCHECK_BZL_FLAG) //tests/unit:test_tensor

graphmemcheck:
	$(GTEST) $(MEMCHECK_BZL_FLAG) //tests/unit:test_graph

operatememcheck:
	$(GTEST) $(MEMCHECK_BZL_FLAG) //tests/unit:test_operate

serializememcheck:
	$(GTEST) $(MEMCHECK_BZL_FLAG) //tests/unit:test_serialize



unittest: tensortest graphtest operatetest serializetest

tensortest:
	$(GTEST) $(REP_BZL_FLAG) //tests/unit:test_tensor

graphtest:
	$(GTEST) $(REP_BZL_FLAG) //tests/unit:test_graph

operatetest:
	$(GTEST) $(REP_BZL_FLAG) //tests/unit:test_operate

serializetest:
	$(GTEST) $(REP_BZL_FLAG) //tests/unit:test_serialize



protodata:
	$(RUN) //tests/py:protogen -- $(shell pwd)/tests/unit/samples

pytest:
	$(TEST) //tests/py:test



accepttest:
	$(GTEST) //tests/regress:test

acceptdata: cleandata
	python tests/regress/tf_generate/tf_generate.py

fmt:
	astyle --project --recursive --suffix=none *.hpp,*.ipp,*.cpp



cleandata:
	rm -f tests/regress/samples/*
	rm -f tests/samples/*
