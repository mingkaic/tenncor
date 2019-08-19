COVERAGE_INFO_FILE := bazel-out/_coverage/_coverage_report.dat

COVERAGE_IGNORE := 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' 'dbg/*'

CCOVER := bazel coverage --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" --config gtest --config cc_coverage

ADE_TEST := //ade:test

TAG_TEST := //tag:test

PBM_TEST := //pbm:test

OPT_TEST := //opt/...

EAD_CTEST := //ead:ctest

CC := gcc

COVERAGE_PIPE := ./bazel-bin/external/com_github_mingkaic_cppkg/merge_cov $(COVERAGE_INFO_FILE)

TMP_LOGFILE := /tmp/tenncor-test.log

print_vars:
	@echo "CC: " $(CC)

rocnnet_py_build:
	bazel build --config $(CC)_eigen_optimal //rocnnet:rocnnet_py

rocnnet_py_export: bazel-bin/rocnnet/rocnnet.so bazel-bin/ead/tenncor.so bazel-bin/ead/ead.so
	cp -f bazel-bin/rocnnet/rocnnet.so rocnnet/notebooks/rocnnet
	cp -f bazel-bin/ead/*.so rocnnet/notebooks/ead


coverage:
	$(CCOVER) $(ADE_TEST) $(TAG_TEST) $(PBM_TEST) $(OPT_TEST) $(EAD_CTEST)
	lcov --remove $(COVERAGE_INFO_FILE) -o coverage.info

cover_ade:
	$(CCOVER) $(ADE_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) -o coverage.info

cover_tag:
	$(CCOVER) $(TAG_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'ade/*' -o coverage.info

cover_pbm:
	$(CCOVER) $(PBM_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'ade/*' -o coverage.info

cover_opt:
	$(CCOVER) $(OPT_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'ade/*' 'tag/*' 'ead/*' -o coverage.info

cover_ead:
	$(CCOVER) $(EAD_CTEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'ade/*' 'tag/*' 'opt/*' -o coverage.info


# optimized comparisons
compare_matmul:
	bazel run $(EIGEN_OPT) //rocnnet:comparison_matmul

compare_mlp:
	bazel run $(EIGEN_OPT) //rocnnet:comparison_mlp

compare_mlp_grad:
	bazel run $(EIGEN_OPT) //rocnnet:comparison_mlp_grad


cov_clean: coverage.info
	lcov --remove coverage.info $(COVERAGE_IGNORE) -o coverage.info
	lcov --list coverage.info

cov_genhtml: coverage.info
	genhtml -o html coverage.info

lcov: coverage cov_clean

lcov_ade: cover_ade cov_clean

lcov_tag: cover_tag cov_clean

lcov_pbm: cover_pbm cov_clean

lcov_opt: cover_opt cov_clean

lcov_ead: cover_ead cov_clean

.PHONY: print_vars rocnnet_py_build rocnnet_py_export
.PHONY: coverage cover_ade cover_tag cover_tag cover_pbm cover_opt cover_ead
.PHONY: compare_matmul compare_mlp compare_mlp_grad cov_clean cov_genhtml
.PHONY: lcov lcov_ade lcov_tag lcov_pbm lcov_opt lcov_ead brief_lcov
