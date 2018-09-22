GTEST_REPEAT := 50

COMMON_BZL_FLAGS := --test_output=all --cache_test_results=no

GTEST_FLAGS := --action_env="GTEST_SHUFFLE=1" --action_env="GTEST_BREAK_ON_FAILURE=1"

REP_BZL_FLAGS := --action_env="GTEST_REPEAT=$(GTEST_REPEAT)"

VAL_BZL_FLAGS := --run_under="valgrind --leak-check=full" --action_env="GTEST_REPEAT=5"

ASAN_BZL_FLAGS := --linkopt -fsanitize=address

BUILD := bazel build

RUN := bazel run

TEST := bazel test $(COMMON_BZL_FLAGS)

GTEST := $(TEST) $(GTEST_FLAGS)

COVER := bazel coverage $(COMMON_BZL_FLAGS) $(GTEST_FLAGS)

COVERAGE_INFO_FILE := coverage.info

# all tests

test: test_util test_ade test_llo check_cli

test_util:
	$(GTEST) //util:test

test_ade: test_ade_dynamic test_ade_static

test_ade_dynamic:
	$(GTEST) $(REP_BZL_FLAGS) //ade:test_dynamic

test_ade_static:
	$(GTEST) //ade:test_static

test_llo:
	$(GTEST) $(REP_BZL_FLAGS) //llo:test

# valgrind unit tests

valgrind: valgrind_util valgrind_ade

valgrind_util:
	$(GTEST) $(VAL_BZL_FLAGS) //util:test

valgrind_ade: valgrind_ade_dynamic valgrind_ade_static

valgrind_ade_dynamic:
	$(GTEST) $(VAL_BZL_FLAGS) //ade:test_dynamic

valgrind_ade_static:
	$(GTEST) $(VAL_BZL_FLAGS) //ade:test_static

# asan unit tests
asan: asan_util asan_ade

asan_util:
	$(GTEST) $(ASAN_BZL_FLAGS) //util:test

asan_ade: asan_ade_dynamic asan_ade_static

asan_ade_dynamic:
	$(GTEST) $(ASAN_BZL_FLAGS) $(REP_BZL_FLAGS) //ade:test_dynamic

asan_ade_static:
	$(GTEST) $(ASAN_BZL_FLAGS) //ade:test_static

# coverage unit tests
lcov_all:
	bash listcov.sh coverage $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_util:
	bash listcov.sh cover_util $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_ade:
	bash listcov.sh cover_ade $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

coverage: cover_util cover_ade

cover_util:
	$(COVER) //util:test --instrumentation_filter=/util[/:]

cover_ade: cover_ade_dynamic cover_ade_static

cover_ade_dynamic:
	$(COVER) $(REP_BZL_FLAGS) //ade:test_dynamic --instrumentation_filter=/ade[/:],/util[/:]

cover_ade_static:
	$(COVER) //ade:test_static --instrumentation_filter=/ade[/:],/util[/:]

# check CLI
check_cli: check_ade_cli check_llo_cli

check_ade_cli: build_ade_cli
	cli/ade/test/check.sh bazel-bin/cli/ade/cli

check_llo_cli: build_llo_cli
	cli/llo/test/check.sh bazel-bin/cli/llo/cli

# build CLI
build_cli: build_ade_cli build_llo_cli

build_ade_cli:
	bazel build //cli/ade:cli

build_llo_cli:
	bazel build //cli/llo:cli


clean:
	bazel clean
