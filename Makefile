GTEST_REPEAT := 50

COMMON_BZL_FLAGS := --test_output=all --cache_test_results=no

GTEST_FLAGS := --action_env="GTEST_SHUFFLE=1" --action_env="GTEST_BREAK_ON_FAILURE=1"

REP_BZL_FLAG := --action_env="GTEST_REPEAT=$(GTEST_REPEAT)"

VAL_BZL_FLAG := --run_under="valgrind --leak-check=full" --action_env="GTEST_REPEAT=5"

ASAN_BZL_FLAG := --linkopt -fsanitize=address

BUILD := bazel build

RUN := bazel run

TEST := bazel test $(COMMON_BZL_FLAGS)

GTEST := $(TEST) $(GTEST_FLAGS)

COVER := bazel coverage $(COMMON_BZL_FLAGS) $(GTEST_FLAGS)

COVERAGE_INFO_FILE := coverage.info

test: test_util test_ade check_cli

test_util:
	$(GTEST) //util:test

test_ade:
	$(GTEST) //ade:test

# valgrind unit tests

valgrind: valgrind_util valgrind_ade

valgrind_util:
	$(GTEST) $(VAL_BZL_FLAG) //util:test

valgrind_ade:
	$(GTEST) $(VAL_BZL_FLAG) //ade:test

# asan unit tests
asan: asan_util asan_ade

asan_util:
	$(GTEST) $(ASAN_BZL_FLAG) //ade:test

asan_ade:
	$(GTEST) $(ASAN_BZL_FLAG) //ade:test

# coverage unit tests
coverage: cover_ade

lcov_util:
	bash listcov.sh cover_ade $(COVERAGE_INFO_FILE)
	lcov --remove $(COVERAGE_INFO_FILE) -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_ade:
	bash listcov.sh cover_ade $(COVERAGE_INFO_FILE)
	lcov --remove $(COVERAGE_INFO_FILE) '**/util/*' -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

cover_util:
	$(COVER) $(REP_BZL_FLAG) //util:test --instrumentation_filter=/util[/:]

cover_ade:
	$(COVER) $(REP_BZL_FLAG) //ade:test --instrumentation_filter=/ade[/:],/util[/:]

# CLI test
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
