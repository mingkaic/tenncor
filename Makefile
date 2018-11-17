GTEST_REPEAT := 50

COVERAGE_INFO_FILE := coverage.info

ERR_TEST := //err:test

ADE_TEST := //ade:test

BWD_TEST := //bwd:test

AGE_TEST := //age:test

AGE_CTEST := //age:ctest


COMMON_BZL_FLAGS := --test_output=all --cache_test_results=no

GTEST_FLAGS := --action_env="GTEST_SHUFFLE=1" --action_env="GTEST_BREAK_ON_FAILURE=1"

REP_BZL_FLAGS := --action_env="GTEST_REPEAT=$(GTEST_REPEAT)"

VALGRIND_CMD := valgrind --leak-check=full

VAL_BZL_FLAGS := --run_under="$(VALGRIND_CMD)"

ASAN_BZL_FLAGS := --linkopt -fsanitize=address

COVERAGE_IGNORE := 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' 'dbg/*'


BUILD := bazel build

RUN := bazel run

TEST := bazel test $(COMMON_BZL_FLAGS)

GTEST := $(TEST) $(GTEST_FLAGS)

COVER := bazel coverage --test_output=all $(GTEST_FLAGS) # we want cache result for coverage

COVERAGE_PIPE := ./scripts/merge_cov.sh $(COVERAGE_INFO_FILE)

TMP_LOGFILE := /tmp/tenncor-test.log

all: test

# all tests

test: test_err test_ade test_age test_bwd test_cage

test_err:
	$(GTEST) $(ERR_TEST)

test_ade:
	$(GTEST) $(REP_BZL_FLAGS) $(ADE_TEST)

test_bwd:
	$(GTEST) $(BWD_TEST)

test_age:
	$(TEST) $(AGE_TEST)

test_cage:
	$(GTEST) $(AGE_CTEST)

# valgrind unit tests

valgrind: valgrind_err valgrind_ade valgrind_bwd valgrind_cage

valgrind_err:
	$(GTEST) $(VAL_BZL_FLAGS) $(ERR_TEST)

valgrind_ade:
	$(GTEST) $(VAL_BZL_FLAGS) --action_env="GTEST_REPEAT=5" $(ADE_TEST)

valgrind_bwd:
	$(GTEST) $(VAL_BZL_FLAGS) $(BWD_TEST)

valgrind_cage:
	$(GTEST) $(VAL_BZL_FLAGS) $(AGE_CTEST)

# asan unit tests

asan: asan_err asan_ade asan_bwd asan_cage

asan_err:
	$(GTEST) $(ASAN_BZL_FLAGS) $(ERR_TEST)

asan_ade:
	$(GTEST) $(ASAN_BZL_FLAGS) $(REP_BZL_FLAGS) $(ADE_TEST)

asan_bwd:
	$(GTEST) $(ASAN_BZL_FLAGS) $(BWD_TEST)

asan_cage:
	$(GTEST) $(ASAN_BZL_FLAGS) $(AGE_CTEST)

# coverage unit tests

coverage: cover_err cover_ade cover_bwd

cover_err:
	$(COVER) --instrumentation_filter= $(ERR_TEST)

cover_ade:
	$(COVER) $(REP_BZL_FLAGS) --instrumentation_filter= $(ADE_TEST)

cover_bwd:
	$(COVER) $(BWD_TEST)

# generated coverage files

lcov_all: coverage
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/err/test/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/ade/test/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/bwd/test/test.log >> $(TMP_LOGFILE)
	cat $(TMP_LOGFILE) | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_err: cover_err
	cat bazel-testlogs/err/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_ade: cover_ade
	cat bazel-testlogs/ade/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) 'log/*' -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_bwd: cover_bwd
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/bwd/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) 'log/*' 'ade/*' -o $(COVERAGE_INFO_FILE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

# deployment

docs:
	rm -rf docs
	doxygen
	mv doxout/html docs
	rm -rf doxout
