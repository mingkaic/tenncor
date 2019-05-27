COVERAGE_INFO_FILE := coverage.info

COVER := bazel coverage --config asan --config gtest

ADE_TEST := //ade:test

EAD_CTEST := //ead:ctest

OPT_TEST := //opt:test

PBM_TEST := //pbm:test

COVERAGE_IGNORE := 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' 'dbg/*'

COVERAGE_PIPE := ./bazel-bin/external/com_github_mingkaic_cppkg/merge_cov $(COVERAGE_INFO_FILE)

TMP_LOGFILE := /tmp/tenncor-test.log

CC := gcc

.PHONY: print_vars
print_vars:
	@echo "CC: " $(CC)

.PHONY: rocnnet_py_build
rocnnet_py_build:
	bazel build --config $(CC)_eigen_optimal //rocnnet:rocnnet_py

.PHONY: rocnnet_py_export
rocnnet_py_export: rocnnet_py_build
	cp -f bazel-bin/rocnnet/*.so rocnnet/notebooks/rocnnet
	cp -f bazel-bin/ead/*.so rocnnet/notebooks/ead


.PHONY: coverage
coverage: cover_ade cover_ead cover_opt cover_pbm

.PHONY: cover_ade
cover_ade:
	$(COVER) $(ADE_TEST)

.PHONY: cover_llo
cover_llo:
	$(COVER) $(LLO_CTEST)

.PHONY: cover_ead
cover_ead:
	$(COVER) $(EAD_CTEST)

.PHONY: cover_opt
cover_opt:
	$(COVER) $(OPT_TEST)

.PHONY: cover_pbm
cover_pbm:
	$(COVER) $(PBM_TEST)


# optimized comparisons
.PHONY: compare_matmul
compare_matmul:
	bazel run $(EIGEN_OPT) //rocnnet:comparison_matmul

.PHONY: compare_mlp
compare_mlp:
	bazel run $(EIGEN_OPT) //rocnnet:comparison_mlp

.PHONY: compare_mlp_grad
compare_mlp_grad:
	bazel run $(EIGEN_OPT) //rocnnet:comparison_mlp_grad


.PHONY: merge_cov
merge_cov:
	bazel build @com_github_mingkaic_cppkg//:merge_cov

.PHONY: lcov
lcov: merge_cov coverage
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/ade/test/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/opt/test/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/ead/ctest/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/pbm/test/test.log >> $(TMP_LOGFILE)
	cat $(TMP_LOGFILE) | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

.PHONY: lcov_ade
lcov_ade: merge_cov cover_ade
	cat bazel-testlogs/ade/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

.PHONY: lcov_opt
lcov_opt: cover_opt
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/opt/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

.PHONY: lcov_ead
lcov_ead: cover_ead
	cat bazel-testlogs/ead/ctest/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) 'opt/*' -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

.PHONY: lcov_pbm
lcov_pbm: cover_pbm
	cat bazel-testlogs/pbm/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)
