COVERAGE_INFO_FILE := bazel-out/_coverage/_coverage_report.dat

COVERAGE_IGNORE := 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' '**/mock/*' '**/*.pb.h' '**/*.pb.cc' 'dbg/*' 'dbg/**/*' 'utils/*' 'utils/**/*' 'perf/*' 'perf/**/*'

CCOVER := bazel coverage --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" --config gtest --config cc_coverage

EIGEN_TEST := //eigen:test

ETEQ_CTEST := //tenncor/eteq:ctest

LAYR_TEST := //tenncor/layr:test

MARSH_TEST := //marsh:test

ONNX_TEST := //onnx:test

OPT_TEST := //opt/...

QUERY_TEST := //query:test

TEQ_TEST := //teq:test

IMAGE_REPO := mkaichen
IMAGE_TAG := latest

REMOTE_CACHE := ''


all: build_test_image build_lib_image build_image

push: push_test_image push_lib_image push_image

.PHONY: build_test_image
build_test_image:
	docker build -f Dockerfile.test -t ${IMAGE_REPO}/tenncor-test:${IMAGE_TAG} .

.PHONY: build_lib_image
build_lib_image:
	docker build -f Dockerfile.make --build-arg REMOTE_CACHE=${REMOTE_CACHE} -t ${IMAGE_REPO}/tenncor-build:${IMAGE_TAG} .
	docker tag ${IMAGE_REPO}/tenncor-build:${IMAGE_TAG} ${IMAGE_REPO}/tenncor-build:latest

.PHONY: build_image
build_image: build_lib_image
	docker build -f Dockerfile -t ${IMAGE_REPO}/tenncor:${IMAGE_TAG} .

.PHONY: push_test_image
push_test_image:
	docker push ${IMAGE_REPO}/tenncor-test:${IMAGE_TAG}

.PHONY: push_lib_image
push_lib_image:
	docker push ${IMAGE_REPO}/tenncor-build:${IMAGE_TAG}

.PHONY: push_image
push_image:
	docker push ${IMAGE_REPO}/tenncor:${IMAGE_TAG}


.PHONY: onnx2json
onnx2json: onnx_test_o2j eteq_test_o2j gd_model_o2j rbm_model_o2j dqn_model_o2j dbn_model_o2j rnn_model_o2j lstm_model_o2j gru_model_o2j

.PHONY: onnx_test_o2j
onnx_test_o2j: models/test/onnx.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/test/onnx.onnx --write /tmp/onnx.json
	mv /tmp/onnx.json models/test

.PHONY: eteq_test_o2j
eteq_test_o2j: models/test/eteq.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/test/eteq.onnx --write /tmp/eteq.json
	mv /tmp/eteq.json models/test

.PHONY: gd_model_o2j
gd_model_o2j: models/gd.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/gd.onnx --write /tmp/gd.json
	mv /tmp/gd.json models

.PHONY: rbm_model_o2j
rbm_model_o2j: models/rbm.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/rbm.onnx --write /tmp/rbm.json
	mv /tmp/rbm.json models

.PHONY: dqn_model_o2j
dqn_model_o2j: models/dqn.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/dqn.onnx --write /tmp/dqn.json
	mv /tmp/dqn.json models

.PHONY: dbn_model_o2j
dbn_model_o2j: models/dbn.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/dbn.onnx --write /tmp/dbn.json
	mv /tmp/dbn.json models

.PHONY: rnn_model_o2j
rnn_model_o2j: models/rnn.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/rnn.onnx --write /tmp/rnn.json
	mv /tmp/rnn.json models

.PHONY: lstm_model_o2j
lstm_model_o2j: models/fast_lstm.onnx models/latin_lstm.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/fast_lstm.onnx --write /tmp/fast_lstm.json
	mv /tmp/fast_lstm.json models
	bazel run //onnx:inspector -- --read ${CURDIR}/models/latin_lstm.onnx --write /tmp/latin_lstm.json
	mv /tmp/latin_lstm.json models

.PHONY: gru_model_o2j
gru_model_o2j: models/fast_gru.onnx models/latin_gru.onnx
	bazel run //onnx:inspector -- --read ${CURDIR}/models/fast_gru.onnx --write /tmp/fast_gru.json
	mv /tmp/fast_gru.json models
	bazel run //onnx:inspector -- --read ${CURDIR}/models/latin_gru.onnx --write /tmp/latin_gru.json
	mv /tmp/latin_gru.json models


.PHONY: json2onnx
json2onnx: onnx_test_j2o eteq_test_j2o gd_model_j2o dqn_model_j2o rbm_model_j2o dbn_model_j2o rnn_model_j2o lstm_model_j2o gru_model_j2o

.PHONY: onnx_test_j2o
onnx_test_j2o: models/test/onnx_graph.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/test/onnx_graph.json --write /tmp/onnx_graph.onnx
	mv /tmp/onnx_graph.onnx models/test

.PHONY: eteq_test_j2o
eteq_test_j2o: models/test/eteq_graph.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/test/eteq_graph.json --write /tmp/eteq_graph.onnx
	mv /tmp/eteq_graph.onnx models/test

.PHONY: gd_model_j2o
gd_model_j2o: models/gd.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/gd.json --write /tmp/gd.onnx
	mv /tmp/gd.onnx models

.PHONY: rbm_model_j2o
rbm_model_j2o: models/rbm.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/rbm.json --write /tmp/rbm.onnx
	mv /tmp/rbm.onnx models

.PHONY: dqn_model_j2o
dqn_model_j2o: models/dqn.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/dqn.json --write /tmp/dqn.onnx
	mv /tmp/dqn.onnx models

.PHONY: dbn_model_j2o
dbn_model_j2o: models/dbn.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/dbn.json --write /tmp/dbn.onnx
	mv /tmp/dbn.onnx models

.PHONY: rnn_model_j2o
rnn_model_j2o: models/rnn.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/rnn.json --write /tmp/rnn.onnx
	mv /tmp/rnn.onnx models

.PHONY: lstm_model_j2o
lstm_model_j2o: models/fast_lstm.json model/latin_lstm.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/fast_lstm.json --write /tmp/fast_lstm.onnx
	mv /tmp/fast_lstm.onnx models
	bazel run //onnx:inspector -- --read ${CURDIR}/models/latin_lstm.json --write /tmp/latin_lstm.onnx
	mv /tmp/latin_lstm.onnx models

.PHONY: gru_model_j2o
gru_model_j2o: models/fast_gru.json model/latin_gru.json
	bazel run //onnx:inspector -- --read ${CURDIR}/models/fast_gru.json --write /tmp/fast_gru.onnx
	mv /tmp/fast_gru.onnx models
	bazel run //onnx:inspector -- --read ${CURDIR}/models/latin_gru.json --write /tmp/latin_gru.onnx
	mv /tmp/latin_gru.onnx models


.PHONY: cov_clean
cov_clean: coverage.info
	lcov --remove coverage.info $(COVERAGE_IGNORE) -o coverage.info
	lcov --list coverage.info

.PHONY: cov_genhtml
cov_genhtml: coverage.info
	genhtml -o html coverage.info


.PHONY: coverage
coverage:
	$(CCOVER) $(EIGEN_TEST) $(ETEQ_CTEST) $(LAYR_TEST) $(ONNX_TEST) $(OPT_TEST) $(QUERY_TEST) $(TEQ_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) -o coverage.info

.PHONY: cover_eigen
cover_eigen:
	$(CCOVER) $(EIGEN_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'onnx/*' 'opt/*' -o coverage.info

.PHONY: cover_eteq
cover_eteq:
	$(CCOVER) $(ETEQ_CTEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'onnx/*' 'opt/*' 'eigen/*' -o coverage.info

.PHONY: cover_layr
cover_layr:
	$(CCOVER) $(LAYR_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'onnx/*' 'opt/*' 'eigen/*' 'eteq/*' -o coverage.info

.PHONY: cover_marsh
cover_marsh:
	$(CCOVER) $(MARSH_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) -o coverage.info

.PHONY: cover_onnx
cover_onnx:
	$(CCOVER) $(ONNX_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' -o coverage.info

.PHONY: cover_opt
cover_opt:
	$(CCOVER) $(OPT_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'onnx/*' 'eigen/*' 'eteq/*' -o coverage.info

.PHONY: cover_query
cover_query:
	$(CCOVER) $(QUERY_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' 'teq/*' 'onnx/*' 'opt/*' 'eigen/*' -o coverage.info

.PHONY: cover_teq
cover_teq:
	$(CCOVER) $(TEQ_TEST)
	lcov --remove $(COVERAGE_INFO_FILE) 'marsh/*' -o coverage.info


.PHONY: lcov
lcov: coverage cov_clean

.PHONY: lcov_eigen
lcov_eigen: cover_eigen cov_clean

.PHONY: lcov_eteq
lcov_eteq: cover_eteq cov_clean

.PHONY: lcov_layr
lcov_layr: cover_layr cov_clean

.PHONY: lcov_onnx
lcov_onnx: cover_onnx cov_clean

.PHONY: lcov_opt
lcov_opt: cover_opt cov_clean

.PHONY: lcov_query
lcov_query: cover_query cov_clean

.PHONY: lcov_teq
lcov_teq: cover_teq cov_clean

.PHONY: install_test_deps
install_test_deps:
	pip3 install -r requirements.txt
	pip3 install -r requirements.test.txt

.PHONY: generate_testcases
generate_testcases:
	bazel run //testutil:tf_gen -- /tmp/tf_testcases.json
	mv /tmp/tf_testcases.json models/test

.PHONY: test_consul_up
test_consul_up:
	docker run -d --name=test-consul -e CONSUL_BIND_INTERFACE=eth0 -p 8500:8500 consul

.PHONY: test_consul_down
test_consul_down:
	docker rm -f test-consul

.PHONY: test_consul_restart
test_consul_restart: test_consul_down test_consul_up
