GTEST_REPEAT := 50

COMMON_BZL_FLAGS := --test_output=all --cache_test_results=no \
	--action_env="GTEST_SHUFFLE=1" --action_env="GTEST_BREAK_ON_FAILURE=1"

REP_BZL_FLAG := --action_env="GTEST_REPEAT=$(GTEST_REPEAT)"

MEMCHECK_BZL_FLAG := --run_under="valgrind"

COVERAGE_BZL_FLAG := --instrumentation_filter= --spawn_strategy=standalone

TEST := bazel test $(COMMON_BZL_FLAGS)

COVER := bazel coverage $(COMMON_BZL_FLAGS) $(COVERAGE_BZL_FLAG)

all: unittest accepttest



coverage: tensorcover graphcover operatecover

tensorcover:
	$(COVER) //tests/unit:test_tensor

graphcover:
	$(COVER) //tests/unit:test_graph

operatecover:
	$(COVER) //tests/unit:test_operate



memcheck: tensortest graphtest operatetest

tensormemcheck:
	$(TEST) $(MEMCHECK_BZL_FLAG) //tests/unit:test_tensor

graphmemcheck:
	$(TEST) $(MEMCHECK_BZL_FLAG) //tests/unit:test_graph

operatememcheck:
	$(TEST) $(MEMCHECK_BZL_FLAG) //tests/unit:test_operate



unittest: tensortest graphtest operatetest

tensortest:
	$(TEST) $(REP_BZL_FLAG) //tests/unit:test_tensor

graphtest:
	$(TEST) $(REP_BZL_FLAG) //tests/unit:test_graph

operatetest:
	$(TEST) $(REP_BZL_FLAG) //tests/unit:test_operate



accepttest:
	$(TEST) //tests/regress:test

acceptdata: cleandata
	python tests/regress/tf_generate/tf_generate.py

fmt:
	astyle --project --recursive --suffix=none *.hpp,*.ipp,*.cpp



cleandata:
	rm -f tests/regress/samples/*
