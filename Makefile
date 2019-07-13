COVERAGE_INFO_FILE := bazel-out/_coverage/_coverage_report.dat

COVERAGE_IGNORE := 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' 'dbg/*'

CCOVER := bazel coverage --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" --config gtest --config cc_coverage

ADE_TEST := //ade:test

TAG_TEST := //tag:test

PBM_TEST := //pbm:test

OPT_TEST := //opt/...

EAD_CTEST := //ead:ctest

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
coverage:
	$(CCOVER) //...

.PHONY: cover_ade
cover_ade:
	$(CCOVER) $(ADE_TEST)

.PHONY: cover_tag
cover_tag:
	$(CCOVER) $(TAG_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'ade/*'

.PHONY: cover_pbm
cover_pbm:
	$(CCOVER) $(PBM_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'ade/*'

.PHONY: cover_opt
cover_opt:
	$(CCOVER) $(OPT_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'ade/*' 'tag/*' 'ead/*'

.PHONY: cover_ead
cover_ead:
	$(CCOVER) $(EAD_CTEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'ade/*' 'tag/*' 'opt/*'


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


.PHONY: cov_clean
cov_clean: $(COVERAGE_INFO_FILE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o coverage.info
	lcov --list coverage.info

.PHONY: cov_genhtml
cov_genhtml: coverage.info
	genhtml -o html coverage.info

.PHONY: lcov
lcov: coverage cov_clean cov_genhtml

.PHONY: lcov_ade
lcov_ade: cover_ade cov_clean cov_genhtml

.PHONY: lcov_tag
lcov_tag: cover_tag cov_clean cov_genhtml

.PHONY: lcov_pbm
lcov_pbm: cover_pbm cov_clean cov_genhtml

.PHONY: lcov_opt
lcov_opt: cover_opt cov_clean cov_genhtml

.PHONY: lcov_ead
lcov_ead: cover_ead cov_clean cov_genhtml
