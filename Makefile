GTEST_REPEAT := 50

COVERAGE_INFO_FILE := coverage.info

ERR_TEST := //err:test

ADE_TEST := //ade:test

AGE_DTEST := //adhoc/age:test_dynamic

AGE_STEST := //adhoc/age:test_static

LLO_TEST := //adhoc/llo:test

REGRESS_TEST := //adhoc/llo:test_regress

PBM_TEST := //pbm:test


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


build_agen:
	bazel build //age:agen


# all tests

test: test_err test_ade test_llo test_pbm

test_err:
	$(GTEST) $(ERR_TEST)

test_ade:
	$(GTEST) $(REP_BZL_FLAGS) $(ADE_TEST)

test_age: test_age_dynamic test_age_static

test_age_dynamic:
	$(GTEST) $(REP_BZL_FLAGS) $(AGE_DTEST)

test_age_static:
	$(GTEST) $(AGE_STEST)

test_llo:
	$(GTEST) $(REP_BZL_FLAGS) $(LLO_TEST)

test_pbm:
	$(GTEST) $(PBM_TEST)

# valgrind unit tests

valgrind: valgrind_err valgrind_ade valgrind_llo valgrind_pbm

valgrind_err:
	$(GTEST) $(VAL_BZL_FLAGS) $(ERR_TEST)

valgrind_ade:
	$(GTEST) $(VAL_BZL_FLAGS) --action_env="GTEST_REPEAT=5" $(ADE_TEST)

valgrind_age: valgrind_age_dynamic valgrind_age_static

valgrind_age_dynamic:
	$(GTEST) $(VAL_BZL_FLAGS) --action_env="GTEST_REPEAT=5" $(AGE_DTEST)

valgrind_age_static:
	$(GTEST) $(VAL_BZL_FLAGS) $(AGE_STEST)

valgrind_llo:
	$(GTEST) $(VAL_BZL_FLAGS) --action_env="GTEST_REPEAT=5" $(LLO_TEST)

valgrind_pbm:
	$(GTEST) $(VAL_BZL_FLAGS) $(PBM_TEST)

# asan unit tests

asan: asan_err asan_ade asan_llo asan_pbm

asan_err:
	$(GTEST) $(ASAN_BZL_FLAGS) $(ERR_TEST)

asan_ade:
	$(GTEST) $(ASAN_BZL_FLAGS) $(REP_BZL_FLAGS) $(ADE_TEST)

asan_age: asan_age_dynamic asan_age_static

asan_age_dynamic:
	$(GTEST) $(ASAN_BZL_FLAGS) $(REP_BZL_FLAGS) $(AGE_DTEST)

asan_age_static:
	$(GTEST) $(ASAN_BZL_FLAGS) $(AGE_STEST)

asan_llo:
	$(GTEST) $(ASAN_BZL_FLAGS) $(REP_BZL_FLAGS) $(LLO_TEST)

asan_pbm:
	$(GTEST) $(ASAN_BZL_FLAGS) $(PBM_TEST)

# coverage unit tests

coverage: cover_err cover_ade cover_llo cover_pbm

cover_err:
	$(COVER) --instrumentation_filter= $(ERR_TEST)

cover_ade:
	$(COVER) $(REP_BZL_FLAGS) --instrumentation_filter= $(ADE_TEST)

cover_age: cover_age_dynamic cover_age_static

cover_ade_dynamic:
	$(COVER) $(REP_BZL_FLAGS) --instrumentation_filter= $(AGE_TEST)

cover_age_static:
	$(COVER) --instrumentation_filter= $(AGE_STEST)

cover_llo:
	$(COVER) $(REP_BZL_FLAGS) --instrumentation_filter= $(LLO_TEST)

cover_pbm:
	$(COVER) --instrumentation_filter= $(PBM_TEST)

# generate stuff

gen_llo: build_agen
	./bazel-bin/age/agen --cfg llo/cfg/llo.json --out llo/generated

lcov_all: coverage
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/err/test/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/ade/test/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/age/test_dynamic/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/age/test_static/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/llo/test/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/pbm/test/test.log >> $(TMP_LOGFILE)
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

lcov_age: cover_age
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/age/test_dynamic/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/age/test_static/test.log >> $(TMP_LOGFILE)
	cat $(TMP_LOGFILE) | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) 'log/*' 'ade/*' -o $(COVERAGE_INFO_FILE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_llo: cover_llo
	cat bazel-testlogs/llo/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) 'log/*' 'ade/*' 'age/*' -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_pbm: cover_pbm
	cat bazel-testlogs/pbm/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) 'log/*' 'ade/*' 'age/*' 'llo/*' -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

# test management

dora_run:
	./scripts/start_dora.sh ./certs

gen_test: dora_run
	bazel run //test_gen:tfgen

test_regress: gen_test
	$(GTEST) $(REGRESS_TEST)

# deployment

docs:
	rm -rf docs
	doxygen
	mv doxout/html docs
	rm -rf doxout
