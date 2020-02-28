COVERAGE_INFO_FILE := bazel-out/_coverage/_coverage_report.dat

COVERAGE_IGNORE := 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' 'dbg/*' '**/mock/*'

CCOVER := bazel coverage --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" --config gtest --config cc_coverage

EIGEN_TEST := //eigen:test

ETEQ_CTEST := //eteq:ctest

LAYR_TEST := //layr:test

MARSH_TEST := //marsh:test

ONNX_TEST := //onnx:test

OPT_TEST := //opt/...

QUERY_TEST := //query:test

TEQ_TEST := //teq:test

CC := clang


print_vars:
	@echo "CC: " $(CC)

rocnnet_py_build:
	bazel build --config $(CC)_eigen_optimal //:tenncor_py

rocnnet_py_export: rocnnet_py_build
	cp -f bazel-bin/*.so rocnnet/notebooks


.PHONY: protoc
protoc:
	mkdir ./build
	bazel build @com_google_protobuf_custom//:protoc
	cp bazel-bin/external/com_google_protobuf_custom/protoc ./build

.PHONY: gen_proto
gen_proto: protoc
	./build/protoc --cpp_out=. -I . onnx/onnx.proto
	./build/protoc --cpp_out=. -I . query/query.proto
	./build/protoc --cpp_out=. -I . opt/optimize.proto


onnx2json: onnx_test_o2j eteq_test_o2j gd_model_o2j rbm_model_o2j dqn_model_o2j dbn_model_o2j rnn_model_o2j lstm_model_o2j gru_model_o2j

onnx_test_o2j: models/test/onnx.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/test/onnx.onnx --write /tmp/onnx.json
	mv /tmp/onnx.json models/test

eteq_test_o2j: models/test/eteq.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/test/eteq.onnx --write /tmp/eteq.json
	mv /tmp/eteq.json models/test

gd_model_o2j: models/gd.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/gd.onnx --write /tmp/gd.json
	mv /tmp/gd.json models

rbm_model_o2j: models/rbm.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/rbm.onnx --write /tmp/rbm.json
	mv /tmp/rbm.json models

dqn_model_o2j: models/dqn.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/dqn.onnx --write /tmp/dqn.json
	mv /tmp/dqn.json models

dbn_model_o2j: models/dbn.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/dbn.onnx --write /tmp/dbn.json
	mv /tmp/dbn.json models

rnn_model_o2j: models/rnn.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/rnn.onnx --write /tmp/rnn.json
	mv /tmp/rnn.json models

lstm_model_o2j: models/lstm.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/lstm.onnx --write /tmp/lstm.json
	mv /tmp/lstm.json models

gru_model_o2j: models/gru.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/gru.onnx --write /tmp/gru.json
	mv /tmp/gru.json models


json2onnx: onnx_test_j2o eteq_test_j2o gd_model_j2o dqn_model_j2o rbm_model_j2o dbn_model_j2o rnn_model_j2o lstm_model_j2o gru_model_j2o

onnx_test_j2o: models/test/onnx_graph.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/test/onnx_graph.json --write /tmp/onnx_graph.onnx
	mv /tmp/onnx_graph.onnx models/test

eteq_test_j2o: models/test/eteq_graph.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/test/eteq_graph.json --write /tmp/eteq_graph.onnx
	mv /tmp/eteq_graph.onnx models/test

gd_model_j2o: models/gd.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/gd.json --write /tmp/gd.onnx
	mv /tmp/gd.onnx models

rbm_model_j2o: models/rbm.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/rbm.json --write /tmp/rbm.onnx
	mv /tmp/rbm.onnx models

dqn_model_j2o: models/dqn.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/dqn.json --write /tmp/dqn.onnx
	mv /tmp/dqn.onnx models

dbn_model_j2o: models/dbn.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/dbn.json --write /tmp/dbn.onnx
	mv /tmp/dbn.onnx models

rnn_model_j2o: models/rnn.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/rnn.json --write /tmp/rnn.onnx
	mv /tmp/rnn.onnx models

lstm_model_j2o: models/lstm.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/lstm.json --write /tmp/lstm.onnx
	mv /tmp/lstm.onnx models

gru_model_j2o: models/gru.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/gru.json --write /tmp/gru.onnx
	mv /tmp/gru.onnx models


cov_clean: coverage.info
	lcov --remove coverage.info $(COVERAGE_IGNORE) -o coverage.info
	lcov --list coverage.info

cov_genhtml: coverage.info
	genhtml -o html coverage.info


coverage:
	$(CCOVER) $(EIGEN_TEST) $(ETEQ_CTEST) $(LAYR_TEST) $(ONNX_TEST) $(OPT_TEST) $(QUERY_TEST) $(TEQ_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) -o coverage.info

cover_eigen:
	$(CCOVER) $(EIGEN_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'onnx/*' 'opt/*' -o coverage.info

cover_eteq:
	$(CCOVER) $(ETEQ_CTEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'onnx/*' 'opt/*' 'eigen/*' -o coverage.info

cover_layr:
	$(CCOVER) $(LAYR_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'onnx/*' 'opt/*' 'eigen/*' 'eteq/*' -o coverage.info

cover_marsh:
	$(CCOVER) $(MARSH_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) -o coverage.info

cover_onnx:
	$(CCOVER) $(ONNX_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' -o coverage.info

cover_opt:
	$(CCOVER) $(OPT_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'onnx/*' 'eigen/*' 'eteq/*' -o coverage.info

cover_query:
	$(CCOVER) $(QUERY_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'onnx/*' 'opt/*' 'eigen/*' -o coverage.info

cover_teq:
	$(CCOVER) $(TEQ_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' -o coverage.info


lcov: coverage cov_clean

lcov_eigen: cover_eigen cov_clean

lcov_eteq: cover_eteq cov_clean

lcov_layr: cover_layr cov_clean

lcov_onnx: cover_onnx cov_clean

lcov_opt: cover_opt cov_clean

lcov_query: cover_query cov_clean

lcov_teq: cover_teq cov_clean


.PHONY: print_vars rocnnet_py_build rocnnet_py_export cov_clean cov_genhtml
.PHONY: coverage cover_eigen cover_eigen cover_eteq cover_layr cover_opt cover_onnx cover_teq
.PHONY: lcov lcov_eigen lcov_eteq lcov_layr lcov_opt lcov_onnx lcov_query lcov_teq
.PHONY: onnx2json onnx_test_o2j eteq_test_o2j gd_model_o2j rbm_model_o2j dqn_model_o2j dbn_model_o2j rnn_model_o2j lstm_model_o2j gru_model_o2j
.PHONY: json2onnx onnx_test_j2o eteq_test_j2o gd_model_j2o rbm_model_j2o dqn_model_j2o dbn_model_j2o rnn_model_j2o lstm_model_j2o gru_model_j2o
