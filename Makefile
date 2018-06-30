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

# unit test
test: test_clay test_mold test_slip test_kiln

test_clay:
	$(GTEST) $(REP_BZL_FLAG) //clay:test

test_mold:
	$(GTEST) $(REP_BZL_FLAG) //mold:test

test_slip:
	$(GTEST) $(REP_BZL_FLAG) //slip:test

test_kiln:
	$(GTEST) $(REP_BZL_FLAG) //kiln:test

test_lead:
	$(GTEST) $(REP_BZL_FLAG) //lead:test

# valgrind unit tests
valgrind: valg_clay valg_mold valg_slip valg_kiln

valg_clay:
	$(GTEST) $(VAL_BZL_FLAG) //clay:test

valg_mold:
	$(GTEST) $(VAL_BZL_FLAG) //mold:test

valg_slip:
	$(GTEST) $(VAL_BZL_FLAG) //slip:test

valg_kiln:
	$(GTEST) $(VAL_BZL_FLAG) //kiln:test

# asan unit tests
asan: asan_clay asan_mold asan_slip asan_kiln

asan_clay:
	$(GTEST) $(ASAN_BZL_FLAG) --action_env="GTEST_REPEAT=25" //clay:test

asan_mold:
	$(GTEST) $(ASAN_BZL_FLAG) --action_env="GTEST_REPEAT=25" //mold:test

asan_slip:
	$(GTEST) $(ASAN_BZL_FLAG) --action_env="GTEST_REPEAT=25" //slip:test

asan_kiln:
	$(GTEST) $(ASAN_BZL_FLAG) --action_env="GTEST_REPEAT=25" //kiln:test

# coverage unit tests
coverage: cover_clay cover_mold cover_slip cover_kiln

lcov_clay:
	bash listcov.sh cover_clay $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_mold:
	bash listcov.sh cover_mold $(COVERAGE_INFO_FILE)
	lcov --remove $(COVERAGE_INFO_FILE) '**/clay/*' -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_slip:
	bash listcov.sh cover_slip $(COVERAGE_INFO_FILE)
	lcov --remove $(COVERAGE_INFO_FILE) '**/clay/*' '**/mold/*' -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_kiln:
	bash listcov.sh cover_kiln $(COVERAGE_INFO_FILE)
	lcov --remove $(COVERAGE_INFO_FILE) '**/clay/*' '**/mold/*' '**/slip/*' -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

cover_clay:
	$(COVER) $(REP_BZL_FLAG) //clay:test --instrumentation_filter=/clay[/:]

cover_mold:
	$(COVER) $(REP_BZL_FLAG) //mold:test --instrumentation_filter=/clay[/:],/mold[/:]

cover_slip:
	$(COVER) $(REP_BZL_FLAG) //slip:test --instrumentation_filter=/clay[/:],/mold[/:],slip[/:]

cover_kiln:
	$(COVER) $(REP_BZL_FLAG) //kiln:test --instrumentation_filter=/clay[/:],/mold[/:],slip[/:],/kiln[/:]

# regression testing
regression:
	$(GTEST) //regress:test

asan_regress:
	$(GTEST) $(ASAN_BZL_FLAG) //regress:test

# todo: deprecate
regress_data: clean_data
	python regress/tf_generate/tf_generate.py

# remove all test data
clean_data:
	rm -f tests/regress/samples/*
	rm -f tests/unit/samples/*

clean: clean_data
	bazel clean
