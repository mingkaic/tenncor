COVERAGE_INFO_FILE := bazel-out/_coverage/_coverage_report.dat

COVERAGE_IGNORE := 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' 'dbg/*'

CCOVER := bazel coverage --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" --config gtest --config cc_coverage --define ETEQ_CFG=MIN

CCUR_TEST := //ccur:test

ETEQ_CTEST := //eteq:ctest

LAYR_TEST := //layr:test

OPT_TEST := //opt/...

PBM_TEST := //pbm:test

TAG_TEST := //tag:test

TEQ_TEST := //teq:test

CC := gcc

COVERAGE_PIPE := ./bazel-bin/external/com_github_mingkaic_cppkg/merge_cov $(COVERAGE_INFO_FILE)

TMP_LOGFILE := /tmp/tenncor-test.log

print_vars:
	@echo "CC: " $(CC)

rocnnet_py_build:
	bazel build --config $(CC)_eigen_optimal //rocnnet:rocnnet_py

rocnnet_py_export: bazel-bin/rocnnet/rocnnet.so bazel-bin/eteq/tenncor.so bazel-bin/eteq/eteq.so
	cp -f bazel-bin/rocnnet/rocnnet.so rocnnet/notebooks/rocnnet
	cp -f bazel-bin/eteq/*.so rocnnet/notebooks/eteq


model_jsonupdate: model_jsongd model_jsondqn model_jsonrbm model_jsondbn

model_jsongd: models/gdmodel.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/gdmodel.pbx --write /tmp/gdmodel.json
	mv /tmp/gdmodel.json models

model_jsondqn: models/dqnmodel.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/dqnmodel.pbx --write /tmp/dqnmodel.json
	mv /tmp/dqnmodel.json models

model_jsonrbm: models/rbmmodel.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/rbmmodel.pbx --write /tmp/rbmmodel.json
	mv /tmp/rbmmodel.json models

model_jsondbn: models/dbnmodel.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/dbnmodel.pbx --write /tmp/dbnmodel.json
	mv /tmp/dbnmodel.json models


coverage:
	$(CCOVER) $(TEQ_TEST) $(TAG_TEST) $(PBM_TEST) $(OPT_TEST) $(ETEQ_CTEST) $(CCUR_TEST) $(LAYR_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) -o coverage.info

cover_ccur:
	$(CCOVER) $(CCUR_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'teq/*' 'tag/*' 'opt/*' 'eteq/*' -o coverage.info

cover_eteq:
	$(CCOVER) $(ETEQ_CTEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'teq/*' 'tag/*' 'opt/*' -o coverage.info

cover_layr:
	$(CCOVER) $(LAYR_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'teq/*' 'tag/*' 'opt/*' 'eteq/*' -o coverage.info

cover_opt:
	$(CCOVER) $(OPT_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'teq/*' 'tag/*' 'eteq/*' -o coverage.info

cover_pbm:
	$(CCOVER) $(PBM_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'teq/*' -o coverage.info

cover_tag:
	$(CCOVER) $(TAG_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'teq/*' -o coverage.info

cover_teq:
	$(CCOVER) $(TEQ_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) -o coverage.info


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

lcov_ccur: cover_ccur cov_clean

lcov_eteq: cover_eteq cov_clean

lcov_layr: cover_layr cov_clean

lcov_opt: cover_opt cov_clean

lcov_pbm: cover_pbm cov_clean

lcov_tag: cover_tag cov_clean

lcov_teq: cover_teq cov_clean

.PHONY: print_vars rocnnet_py_build rocnnet_py_export
.PHONY: coverage cover_ccur cover_eteq cover_layr cover_opt cover_pbm cover_tag cover_teq
.PHONY: compare_matmul compare_mlp compare_mlp_grad cov_clean cov_genhtml
.PHONY: lcov lcov_ccur lcov_eteq lcov_layr lcov_opt lcov_pbm lcov_tag lcov_teq
.PHONY: model_jsonupdate model_jsongd model_jsondqn model_jsonrbm model_jsondbn
