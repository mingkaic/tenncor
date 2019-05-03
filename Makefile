COVERAGE_INFO_FILE := coverage.info

COVER := bazel coverage --config asan --config gtest

ADE_TEST := //ade:test

BWD_TEST := //bwd:test

LLO_CTEST := //llo:ctest

LLO_PTEST := //llo:ptest

OPT_TEST := //opt:test

PBM_TEST := //pbm:test

COVERAGE_IGNORE := 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' 'dbg/*'

COVERAGE_PIPE := ./bazel-bin/external/com_github_mingkaic_cppkg/merge_cov $(COVERAGE_INFO_FILE)

TMP_LOGFILE := /tmp/tenncor-test.log

EIGEN_OPT = --copt="-openmp" --copt="-march=native" --copt="-O3"


rocnnet_py_build:
	bazel build $(EIGEN_OPT) //rocnnet:rocnnet_py

rocnnet_py_export: rocnnet_py_build
	cp -f bazel-bin/rocnnet/*.so rocnnet/notebooks/rocnnet
	cp -f bazel-bin/ead/*.so rocnnet/notebooks/ead


coverage: cover_ade cover_bwd cover_ead cover_opt cover_pbm

cover_ade:
	$(COVER) $(ADE_TEST)

cover_bwd:
	$(COVER) $(BWD_TEST)

cover_llo:
	$(COVER) $(LLO_CTEST)

cover_ead:
	$(COVER) $(EAD_CTEST)

cover_opt:
	$(COVER) $(OPT_TEST)

cover_pbm:
	$(COVER) $(PBM_TEST)


# optimized comparisons
compare_matmul:
	bazel run $(EIGEN_OPT) //rocnnet:comparison_matmul

compare_mlp:
	bazel run $(EIGEN_OPT) //rocnnet:comparison_mlp

compare_mlp_grad:
	bazel run $(EIGEN_OPT) //rocnnet:comparison_mlp_grad


merge_cov:
	bazel build @com_github_mingkaic_cppkg//:merge_cov

lcov: merge_cov coverage
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/ade/test/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/bwd/test/test.log >> $(TMP_LOGFILE)
	cat $(TMP_LOGFILE) | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_ade: merge_cov cover_ade
	cat bazel-testlogs/ade/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_bwd: merge_cov over_bwd
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/bwd/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) 'ade/*' -o $(COVERAGE_INFO_FILE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)
