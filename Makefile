COVERAGE_INFO_FILE := bazel-out/_coverage/_coverage_report.dat

COVERAGE_IGNORE := 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' 'dbg/*' '**/mock/*'

CCOVER := bazel coverage --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" --config gtest --config cc_coverage

CCUR_TEST := //ccur:test

EIGEN_TEST := //eigen:test

ETEQ_CTEST := //eteq:ctest

LAYR_TEST := //layr:test

MARSH_TEST := //marsh:test

OPT_TEST := //opt/...

PBM_TEST := //pbm:test

TAG_TEST := //tag:test

TEQ_TEST := //teq:test

CC := clang

COVERAGE_PIPE := ./bazel-bin/external/com_github_mingkaic_cppkg/merge_cov $(COVERAGE_INFO_FILE)

TMP_LOGFILE := /tmp/tenncor-test.log

print_vars:
	@echo "CC: " $(CC)

rocnnet_py_build:
	bazel build --config $(CC)_eigen_optimal //rocnnet:rocnnet_py

rocnnet_py_export: bazel-bin/rocnnet/rocnnet.so bazel-bin/eteq/tenncor.so bazel-bin/eteq/eteq.so
	cp -f bazel-bin/rocnnet/rocnnet.so rocnnet/notebooks/rocnnet
	cp -f bazel-bin/eteq/*.so rocnnet/notebooks/eteq


onnx2json: eteq_test_o2j onnx_test_o2j

eteq_test_o2j: models/test/eteq_graph.onnx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/test/eteq_graph.onnx --write /tmp/eteq_graph.json
	mv /tmp/eteq_graph.json models/test

onnx_test_o2j: models/test/onnx_graph.onnx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/test/onnx_graph.onnx --write /tmp/onnx_graph.json
	mv /tmp/onnx_graph.json models/test


pbx2json: eteq_test_p2j pbm_test_p2j gd_model_p2j dqn_model_p2j rbm_model_p2j dbn_model_p2j rnn_model_p2j lstm_model_p2j gru_model_p2j

eteq_test_p2j: models/test/eteq.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/test/eteq.pbx --write /tmp/eteq.json
	mv /tmp/eteq.json models/test

pbm_test_p2j: models/test/pbm.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/test/pbm.pbx --write /tmp/pbm.json
	mv /tmp/pbm.json models/test

gd_model_p2j: models/gd.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/gd.pbx --write /tmp/gd.json
	mv /tmp/gd.json models

dqn_model_p2j: models/dqn.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/dqn.pbx --write /tmp/dqn.json
	mv /tmp/dqn.json models

rbm_model_p2j: models/rbm.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/rbm.pbx --write /tmp/rbm.json
	mv /tmp/rbm.json models

dbn_model_p2j: models/dbn.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/dbn.pbx --write /tmp/dbn.json
	mv /tmp/dbn.json models

rnn_model_p2j: models/rnn.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/rnn.pbx --write /tmp/rnn.json
	mv /tmp/rnn.json models

lstm_model_p2j: models/lstm.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/lstm.pbx --write /tmp/lstm.json
	mv /tmp/lstm.json models

gru_model_p2j: models/gru.pbx
	bazel run //pbm:inspector -- --read ${CURDIR}/models/gru.pbx --write /tmp/gru.json
	mv /tmp/gru.json models


json2onnx: eteq_test_j2o onnx_test_j2o

eteq_test_j2o: models/test/eteq_graph.json
	bazel run //pbm:inspector -- --read ${CURDIR}/models/test/eteq_graph.json --write /tmp/eteq_graph.onnx
	mv /tmp/eteq_graph.onnx models/test

onnx_test_j2o: models/test/onnx_graph.json
	bazel run //pbm:inspector -- --read ${CURDIR}/models/test/onnx_graph.json --write /tmp/onnx_graph.onnx
	mv /tmp/onnx_graph.onnx models/test


json2pbx: eteq_test_j2p pbm_test_j2p gd_model_j2p dqn_model_j2p rbm_model_j2p dbn_model_j2p rnn_model_j2p lstm_model_j2p gru_model_j2p

eteq_test_j2p: models/test/eteq.json
	bazel run //pbm:inspector -- --read ${CURDIR}/models/test/eteq.json --write /tmp/eteq.pbx
	mv /tmp/eteq.pbx models/test

pbm_test_j2p: models/test/pbm.json
	bazel run //pbm:inspector -- --read ${CURDIR}/models/test/pbm.json --write /tmp/pbm.pbx
	mv /tmp/pbm.pbx models/test

gd_model_j2p: models/gd.json
	bazel run //pbm:inspector -- --read ${CURDIR}/models/gd.json --write /tmp/gd.pbx
	mv /tmp/gd.pbx models

dqn_model_j2p: models/dqn.json
	bazel run //pbm:inspector -- --read ${CURDIR}/models/dqn.json --write /tmp/dqn.pbx
	mv /tmp/dqn.pbx models

rbm_model_j2p: models/rbm.json
	bazel run //pbm:inspector -- --read ${CURDIR}/models/rbm.json --write /tmp/rbm.pbx
	mv /tmp/rbm.pbx models

dbn_model_j2p: models/dbn.json
	bazel run //pbm:inspector -- --read ${CURDIR}/models/dbn.json --write /tmp/dbn.pbx
	mv /tmp/dbn.pbx models

rnn_model_j2p: models/rnn.json
	bazel run //pbm:inspector -- --read ${CURDIR}/models/rnn.json --write /tmp/rnn.pbx
	mv /tmp/rnn.pbx models

lstm_model_j2p: models/lstm.json
	bazel run //pbm:inspector -- --read ${CURDIR}/models/lstm.json --write /tmp/lstm.pbx
	mv /tmp/lstm.pbx models

gru_model_j2p: models/gru.json
	bazel run //pbm:inspector -- --read ${CURDIR}/models/gru.json --write /tmp/gru.pbx
	mv /tmp/gru.pbx models


coverage:
	$(CCOVER) $(TEQ_TEST) $(TAG_TEST) $(PBM_TEST) $(OPT_TEST) $(EIGEN_TEST) $(ETEQ_CTEST) $(CCUR_TEST) $(LAYR_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) -o coverage.info

cover_ccur:
	$(CCOVER) $(CCUR_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'tag/*' 'opt/*' 'eigen/*' 'eteq/*' -o coverage.info

cover_eigen:
	$(CCOVER) $(EIGEN_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'tag/*' 'opt/*' -o coverage.info

cover_eteq:
	$(CCOVER) $(ETEQ_CTEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'tag/*' 'opt/*' 'eigen/*' -o coverage.info

cover_layr:
	$(CCOVER) $(LAYR_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'tag/*' 'opt/*' 'eigen/*' 'eteq/*' -o coverage.info

cover_marsh:
	$(CCOVER) $(MARSH_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) -o coverage.info

cover_opt:
	$(CCOVER) $(OPT_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'tag/*' 'eigen/*' 'eteq/*' -o coverage.info

cover_pbm:
	$(CCOVER) $(PBM_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' -o coverage.info

cover_tag:
	$(CCOVER) $(TAG_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' -o coverage.info

cover_teq:
	$(CCOVER) $(TEQ_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' -o coverage.info


# optimized comparisons
compare_matmul:
	bazel run --config $(CC)_eigen_optimal //rocnnet:comparison_matmul

compare_mlp:
	bazel run --config $(CC)_eigen_optimal //rocnnet:comparison_mlp

compare_mlp_grad:
	bazel run --config $(CC)_eigen_optimal //rocnnet:comparison_mlp_grad


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
.PHONY: coverage cover_ccur cover_eigen cover_eteq cover_layr cover_opt cover_pbm cover_tag cover_teq
.PHONY: compare_matmul compare_mlp compare_mlp_grad cov_clean cov_genhtml
.PHONY: lcov lcov_ccur lcov_eteq lcov_layr lcov_opt lcov_pbm lcov_tag lcov_teq
.PHONY: onnx2json eteq_test_o2j pbm_test_o2j
.PHONY: json2onnx eteq_test_j2o pbm_test_j2o
.PHONY: pbx2json eteq_test_p2j pbm_test_p2j gd_model_p2j dqn_model_p2j rbm_model_p2j dbn_model_p2j rnn_model_p2j lstm_model_p2j gru_model_p2j
.PHONY: json2pbx eteq_test_j2p pbm_test_j2p gd_model_j2p dqn_model_j2p rbm_model_j2p dbn_model_j2p rnn_model_j2p lstm_model_j2p gru_model_j2p
