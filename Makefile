
######## TEST SETUP ########

.PHONY: install_test_deps
install_test_deps:
	pip3 install -r requirements.test.txt

.PHONY: generate_testcases
generate_testcases: install_test_deps
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

######## BUILD AND PUSH DOCKER IMAGES ########

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

######## MANUALLY GENERATE PROTOBUF ########

PROTOC := bazel-bin/external/com_google_protobuf/protoc
GRPC_CPP_PLUGIN := bazel-bin/external/com_github_grpc_grpc/src/compiler/grpc_cpp_plugin

${PROTOC}:
	bazel build @com_google_protobuf//:protoc

${GRPC_CPP_PLUGIN}:
	bazel build @com_github_grpc_grpc//src/compiler:grpc_cpp_plugin

.PHONY: gen-proto
gen-proto: gen-extenncor-proto gen-onnx-proto gen-gemit-proto gen-oxsvc-proto

.PHONY: gen-extenncor-proto
gen-extenncor-proto: ${PROTOC}
	./${PROTOC} --python_out=. -I . extenncor/dataset_trainer.proto extenncor/dqn_trainer.proto

.PHONY: gen-onnx-proto
gen-onnx-proto: ${PROTOC}
	./${PROTOC} --cpp_out=. -I . internal/onnx/onnx.proto

.PHONY: gen-gemit-proto
gen-gemit-proto: ${PROTOC} ${GRPC_CPP_PLUGIN}
	./${PROTOC} --cpp_out=. -I . dbg/peval/emit/gemitter.proto
	./${PROTOC} --grpc_out=. --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN} -I . dbg/peval/emit/gemitter.proto

.PHONY: gen-oxsvc-proto
gen-oxsvc-proto: ${PROTOC} ${GRPC_CPP_PLUGIN}
	./${PROTOC} --cpp_out=. -I . tenncor/serial/oxsvc/distr.ox.proto
	./${PROTOC} --grpc_out=. --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN} -I . tenncor/serial/oxsvc/distr.ox.proto

######## MODEL FILE GENERATION ########

.PHONY: onnx2json
onnx2json: onnx_test_o2j eteq_test_o2j edeps_test_o2j gd_model_o2j rbm_model_o2j dqn_model_o2j dbn_model_o2j rnn_model_o2j lstm_model_o2j gru_model_o2j

.PHONY: onnx_test_o2j
onnx_test_o2j: models/test/simple_onnx.onnx models/test/layer_onnx.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/simple_onnx.onnx --write /tmp/simple_onnx.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/layer_onnx.onnx --write /tmp/layer_onnx.json
	mv /tmp/simple_onnx.json models/test
	mv /tmp/layer_onnx.json models/test

.PHONY: eteq_test_o2j
eteq_test_o2j: models/test/eteq.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/eteq.onnx --write /tmp/eteq.json
	mv /tmp/eteq.json models/test

.PHONY: edeps_test_o2j
edeps_test_o2j: models/test/edeps.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/edeps.onnx --write /tmp/edeps.json
	mv /tmp/edeps.json models/test

.PHONY: gd_model_o2j
gd_model_o2j: models/gd.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/gd.onnx --write /tmp/gd.json
	mv /tmp/gd.json models

.PHONY: rbm_model_o2j
rbm_model_o2j: models/rbm.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/rbm.onnx --write /tmp/rbm.json
	mv /tmp/rbm.json models

.PHONY: dqn_model_o2j
dqn_model_o2j: models/dqn.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/dqn.onnx --write /tmp/dqn.json
	mv /tmp/dqn.json models

.PHONY: dbn_model_o2j
dbn_model_o2j: models/dbn.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/dbn.onnx --write /tmp/dbn.json
	mv /tmp/dbn.json models

.PHONY: rnn_model_o2j
rnn_model_o2j: models/rnn.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/rnn.onnx --write /tmp/rnn.json
	mv /tmp/rnn.json models

.PHONY: lstm_model_o2j
lstm_model_o2j: models/fast_lstm.onnx models/latin_lstm.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/fast_lstm.onnx --write /tmp/fast_lstm.json
	mv /tmp/fast_lstm.json models
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/latin_lstm.onnx --write /tmp/latin_lstm.json
	mv /tmp/latin_lstm.json models

.PHONY: gru_model_o2j
gru_model_o2j: models/fast_gru.onnx models/latin_gru.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/fast_gru.onnx --write /tmp/fast_gru.json
	mv /tmp/fast_gru.json models
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/latin_gru.onnx --write /tmp/latin_gru.json
	mv /tmp/latin_gru.json models

.PHONY: json2onnx
json2onnx: onnx_test_j2o eteq_test_j2o edeps_test_j2o gd_model_j2o dqn_model_j2o rbm_model_j2o dbn_model_j2o rnn_model_j2o lstm_model_j2o gru_model_j2o

.PHONY: onnx_test_j2o
onnx_test_j2o: models/test/simple_onnx.json models/test/layer_onnx.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/simple_onnx.json --write /tmp/simple_onnx.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/layer_onnx.json --write /tmp/layer_onnx.onnx
	mv /tmp/simple_onnx.onnx models/test
	mv /tmp/layer_onnx.onnx models/test

.PHONY: eteq_test_j2o
eteq_test_j2o: models/test/eteq.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/eteq.json --write /tmp/eteq.onnx
	mv /tmp/eteq.onnx models/test

.PHONY: edeps_test_j2o
edeps_test_j2o: models/test/edeps.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/edeps.json --write /tmp/edeps.onnx
	mv /tmp/edeps.onnx models/test

.PHONY: gd_model_j2o
gd_model_j2o: models/gd.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/gd.json --write /tmp/gd.onnx
	mv /tmp/gd.onnx models

.PHONY: rbm_model_j2o
rbm_model_j2o: models/rbm.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/rbm.json --write /tmp/rbm.onnx
	mv /tmp/rbm.onnx models

.PHONY: dqn_model_j2o
dqn_model_j2o: models/dqn.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/dqn.json --write /tmp/dqn.onnx
	mv /tmp/dqn.onnx models

.PHONY: dbn_model_j2o
dbn_model_j2o: models/dbn.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/dbn.json --write /tmp/dbn.onnx
	mv /tmp/dbn.onnx models

.PHONY: rnn_model_j2o
rnn_model_j2o: models/rnn.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/rnn.json --write /tmp/rnn.onnx
	mv /tmp/rnn.onnx models

.PHONY: lstm_model_j2o
lstm_model_j2o: models/fast_lstm.json models/latin_lstm.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/fast_lstm.json --write /tmp/fast_lstm.onnx
	mv /tmp/fast_lstm.onnx models
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/latin_lstm.json --write /tmp/latin_lstm.onnx
	mv /tmp/latin_lstm.onnx models

.PHONY: gru_model_j2o
gru_model_j2o: models/fast_gru.json models/latin_gru.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/fast_gru.json --write /tmp/fast_gru.onnx
	mv /tmp/fast_gru.onnx models
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/latin_gru.json --write /tmp/latin_gru.onnx
	mv /tmp/latin_gru.onnx models

######## COVERAGES ########

LOC_COVERAGE_FILE := coverage.info
COVERAGE_INFO_FILE := bazel-out/_coverage/_coverage_report.dat
CCOVER := bazel coverage --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" --config gtest --config cc_coverage
ETEQ_CTEST := //tenncor/eteq:ctest
LAYR_CTEST := //tenncor/layr:ctest
DISTRIB_CTEST := //tenncor/distrib:ctest

.PHONY: cov_clean
cov_clean: ${LOC_COVERAGE_FILE}
	rm ${LOC_COVERAGE_FILE}
	rm -Rf html

.PHONY: cov_genhtml
cov_genhtml: ${LOC_COVERAGE_FILE}
	genhtml -o html ${LOC_COVERAGE_FILE}

.PHONY: clean_test_coverage
clean_test_coverage: ${COVERAGE_INFO_FILE}
	lcov --remove ${COVERAGE_INFO_FILE} '**/test/*' '**/mock/*' '**/*.pb.*' -o ${LOC_COVERAGE_FILE}

.PHONY: coverage
coverage:
	${CCOVER} //internal/... ${DISTRIB_CTEST} ${ETEQ_CTEST} ${HONE_CTEST} ${LAYR_CTEST} ${SERIAL_CTEST}
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'internal/*' 'tenncor/*' -o ${LOC_COVERAGE_FILE}

###### INDIVIDUAL COVERAGES ######

#### internal coverages ####

.PHONY: cover_global
cover_global:
	${CCOVER} //internal/global:test
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'internal/global/*' -o ${LOC_COVERAGE_FILE}

.PHONY: cover_marsh
cover_marsh:
	${CCOVER} //internal/marsh/...
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'internal/marsh/*' -o ${LOC_COVERAGE_FILE}

.PHONY: cover_teq
cover_teq:
	${CCOVER} //internal/teq/...
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'internal/teq/*' -o ${LOC_COVERAGE_FILE}

.PHONY: cover_eigen
cover_eigen:
	${CCOVER} //internal/eigen/...
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'internal/eigen/*' -o ${LOC_COVERAGE_FILE}

.PHONY: cover_onnx
cover_onnx:
	${CCOVER} //internal/onnx/...
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'internal/onnx/*' -o ${LOC_COVERAGE_FILE}

.PHONY: cover_query
cover_query:
	${CCOVER} //internal/query/...
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'internal/query/*' -o ${LOC_COVERAGE_FILE}

.PHONY: cover_opt
cover_opt:
	${CCOVER} //internal/opt/...
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'internal/opt/*' -o ${LOC_COVERAGE_FILE}

.PHONY: cover_util
cover_util:
	${CCOVER} //internal/util/...
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'internal/util/*' -o ${LOC_COVERAGE_FILE}

#### tenncor coverages ####

.PHONY: cover_distrib
cover_distrib:
	${CCOVER} ${DISTRIB_CTEST}
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'tenncor/distrib/*' -o ${LOC_COVERAGE_FILE}

.PHONY: cover_eteq
cover_eteq:
	${CCOVER} ${ETEQ_CTEST}
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'tenncor/eteq/*' -o ${LOC_COVERAGE_FILE}

.PHONY: cover_hone
cover_hone:
	${CCOVER} ${HONE_CTEST}
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'tenncor/hone/*' -o ${LOC_COVERAGE_FILE}

.PHONY: cover_layr
cover_layr:
	${CCOVER} ${LAYR_CTEST}
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'tenncor/layr/*' -o ${LOC_COVERAGE_FILE}

.PHONY: cover_serial
cover_serial:
	${CCOVER} ${SERIAL_CTEST}
	@make clean_test_coverage
	lcov --extract ${LOC_COVERAGE_FILE} 'tenncor/serial/*' -o ${LOC_COVERAGE_FILE}
