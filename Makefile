COVERAGE_INFO_FILE := coverage.info

COVER := bazel coverage --config asan --config gtest

ERR_TEST := //err:test

ADE_TEST := //ade:test

BWD_TEST := //bwd:test

COVERAGE_IGNORE := 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' 'dbg/*'

COVERAGE_PIPE := ./scripts/merge_cov.sh $(COVERAGE_INFO_FILE)

TMP_LOGFILE := /tmp/tenncor-test.log


coverage: cover_err cover_ade cover_bwd

cover_err:
	$(COVER) $(ERR_TEST)

cover_ade:
	$(COVER) $(ADE_TEST)

cover_bwd:
	$(COVER) $(BWD_TEST)


lcov: coverage
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


docs:
	rm -rf docs
	doxygen
	mv doxout/html docs
	rm -rf doxout
