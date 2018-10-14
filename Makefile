GTEST_REPEAT := 50

COVERAGE_INFO_FILE := coverage.info

UTIL_TEST := //util:test_util

ADE_DTEST := //ade:test_dynamic

ADE_STEST := //ade:test_static

ADE_UTEST := //ade:test_util

LLO_TEST := //llo:test_llo

REGRESS_TEST := //llo:test_regress

PBM_TEST := //pbm:test_pbm


COMMON_BZL_FLAGS := --test_output=all --cache_test_results=no

GTEST_FLAGS := --action_env="GTEST_SHUFFLE=1" --action_env="GTEST_BREAK_ON_FAILURE=1"

REP_BZL_FLAGS := --action_env="GTEST_REPEAT=$(GTEST_REPEAT)"

VALGRIND_CMD := valgrind --leak-check=full

VAL_BZL_FLAGS := --run_under="$(VALGRIND_CMD)"

ASAN_BZL_FLAGS := --linkopt -fsanitize=address

COVERAGE_IGNORE := 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*'


BUILD := bazel build

RUN := bazel run

TEST := bazel test $(COMMON_BZL_FLAGS)

GTEST := $(TEST) $(GTEST_FLAGS)

COVER := bazel coverage --test_output=all $(GTEST_FLAGS) # we want cache result for coverage

COVERAGE_PIPE := ./listcov.sh $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE)

TMP_LOGFILE := /tmp/tenncor-test.log


# all tests

test: test_ade test_llo test_pbm check_cli

test_ade: test_ade_util test_ade_dynamic test_ade_static

test_ade_util:
	$(GTEST) $(ADE_UTEST)

test_ade_dynamic:
	$(GTEST) $(REP_BZL_FLAGS) $(ADE_DTEST)

test_ade_static:
	$(GTEST) $(ADE_STEST)

test_llo:
	$(GTEST) $(REP_BZL_FLAGS) $(LLO_TEST)

test_regress: gen_test
	$(GTEST) $(REGRESS_TEST)

test_pbm:
	$(GTEST) $(PBM_TEST)

# valgrind unit tests

valgrind: valgrind_util valgrind_ade valgrind_llo valgrind_pbm valgrind_cli

valgrind_util:
	$(GTEST) $(VAL_BZL_FLAGS) $(UTIL_TEST)

valgrind_ade: valgrind_ade_dynamic valgrind_ade_static

valgrind_ade_dynamic:
	$(GTEST) $(VAL_BZL_FLAGS) --action_env="GTEST_REPEAT=5" $(ADE_DTEST)

valgrind_ade_static:
	$(GTEST) $(VAL_BZL_FLAGS) $(ADE_STEST)

valgrind_llo:
	$(GTEST) $(VAL_BZL_FLAGS) --action_env="GTEST_REPEAT=5" $(LLO_TEST)

valgrind_pbm:
	$(GTEST) $(VAL_BZL_FLAGS) $(PBM_TEST)

# asan unit tests

asan: asan_util asan_ade asan_llo asan_pbm

asan_util:
	$(GTEST) $(ASAN_BZL_FLAGS) $(UTIL_TEST)

asan_ade: asan_ade_dynamic asan_ade_static

asan_ade_dynamic:
	$(GTEST) $(ASAN_BZL_FLAGS) $(REP_BZL_FLAGS) $(ADE_DTEST)

asan_ade_static:
	$(GTEST) $(ASAN_BZL_FLAGS) $(ADE_STEST)

asan_llo:
	$(GTEST) $(ASAN_BZL_FLAGS) $(REP_BZL_FLAGS) $(LLO_TEST)

asan_pbm:
	$(GTEST) $(ASAN_BZL_FLAGS) $(PBM_TEST)

# coverage unit tests

coverage: cover_util cover_ade cover_llo cover_pbm

cover_util:
	$(COVER) --instrumentation_filter=/util[/:] $(UTIL_TEST)

cover_ade: cover_ade_dynamic cover_ade_static

cover_ade_dynamic:
	$(COVER) $(REP_BZL_FLAGS) --instrumentation_filter=/ade[/:],/util[/:] $(ADE_DTEST)

cover_ade_static:
	$(COVER) --instrumentation_filter=/ade[/:],/util[/:] $(ADE_STEST)

cover_llo:
	$(COVER) $(REP_BZL_FLAGS) --instrumentation_filter=/llo[/:],/ade[/:],/util[/:] $(LLO_TEST)

cover_pbm:
	$(COVER) --instrumentation_filter=/pbm[/:],/llo[/:],/ade[/:],/util[/:] $(PBM_TEST)

# generate coverage.info

lcov_all: coverage
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/util/test_util/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/ade/test_dynamic/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/ade/test_static/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/llo/test_llo/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/pbm/test_pbm/test.log >> $(TMP_LOGFILE)
	cat $(TMP_LOGFILE) | $(COVERAGE_PIPE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_util: cover_util
	cat bazel-testlogs/util/test_util/test.log | $(COVERAGE_PIPE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_ade: cover_ade
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/ade/test_dynamic/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/ade/test_static/test.log >> $(TMP_LOGFILE)
	cat $(TMP_LOGFILE) | $(COVERAGE_PIPE) 'util/*'
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_llo: cover_llo
	cat bazel-testlogs/llo/test_llo/test.log | $(COVERAGE_PIPE) 'util/*' 'ade/*'
	lcov --list $(COVERAGE_INFO_FILE)

lcov_pbm: cover_pbm
	cat bazel-testlogs/pbm/test_pbm/test.log | $(COVERAGE_PIPE) 'util/*' 'ade/*' 'llo/*'
	lcov --list $(COVERAGE_INFO_FILE)

# check CLI

check_cli: check_ade_cli check_llo_cli

check_ade_cli: build_ade_cli
	cli/ade/test/check.sh bazel-bin/cli/ade/cli

check_llo_cli: build_llo_cli
	cli/llo/test/check.sh bazel-bin/cli/llo/cli

# check CLI with valgrind

valgrind_cli: valgrind_ade_cli valgrind_llo_cli

valgrind_ade_cli: build_ade_cli
	cli/ade/test/check.sh $(VALGRIND_CMD) bazel-bin/cli/ade/cli

valgrind_llo_cli: build_llo_cli
	cli/llo/test/check.sh $(VALGRIND_CMD) bazel-bin/cli/llo/cli

# build CLI

build_cli: build_ade_cli build_llo_cli

build_ade_cli:
	bazel build //cli/ade:cli

build_llo_cli:
	bazel build //cli/llo:cli

# test management

dora_run:
	./start_dora.sh

gen_test: dora_run
	bazel run //test_gen:tfgen
