
######## TEST SETUP ########

.PHONY: install_test_deps
install_test_deps:
	pip3 install -r requirements.test.txt

.PHONY: generate_testcases
generate_testcases: install_test_deps
	bazel run //testutil:tf_gen -- /tmp/tf_testcases.json
	mv /tmp/tf_testcases.json models/test/testcases/
	jq -c . < /tmp/tf_testcases.json | split -b 10000000 --additional-suffix=.json - models/test/testcases/tf_cases_

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
gen-proto: gen-gemit-proto gen-extenncor-proto gen-onnx-proto gen-oxsvc-proto

.PHONY: gen-gemit-proto
gen-gemit-proto: ${PROTOC} ${GRPC_CPP_PLUGIN}
	./${PROTOC} --cpp_out=. -I . dbg/peval/emit/gemitter.proto
	./${PROTOC} --grpc_out=. --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN} -I . dbg/peval/emit/gemitter.proto

.PHONY: gen-extenncor-proto
gen-extenncor-proto: ${PROTOC}
	./${PROTOC} --python_out=. -I . extenncor/dataset_trainer.proto extenncor/dqn_trainer.proto

.PHONY: gen-onnx-proto
gen-onnx-proto: ${PROTOC}
	./${PROTOC} --cpp_out=. -I . internal/onnx/onnx.proto

.PHONY: gen-oxsvc-proto
gen-oxsvc-proto: ${PROTOC} ${GRPC_CPP_PLUGIN}
	./${PROTOC} --cpp_out=. -I . tenncor/serial/oxsvc/distr.ox.proto
	./${PROTOC} --grpc_out=. --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN} -I . tenncor/serial/oxsvc/distr.ox.proto

######## MODEL FILE GENERATION ########

.PHONY: onnx2json
onnx2json: onnx_test_o2j eteq_test_o2j serial_test_o2j oxsvc_test_o2j gd_model_o2j rbm_model_o2j dqn_model_o2j dbn_model_o2j rnn_model_o2j lstm_model_o2j gru_model_o2j

.PHONY: onnx_test_o2j
onnx_test_o2j: models/test/bad_onnx.onnx models/test/bad_onnx2.onnx models/test/simple_onnx.onnx models/test/simple_stop.onnx models/test/layer_onnx.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/bad_onnx.onnx --write /tmp/bad_onnx.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/bad_onnx2.onnx --write /tmp/bad_onnx2.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/simple_onnx.onnx --write /tmp/simple_onnx.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/simple_stop.onnx --write /tmp/simple_stop.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/layer_onnx.onnx --write /tmp/layer_onnx.json
	mv /tmp/bad_onnx.json models/test
	mv /tmp/bad_onnx2.json models/test
	mv /tmp/simple_onnx.json models/test
	mv /tmp/simple_stop.json models/test
	mv /tmp/layer_onnx.json models/test

.PHONY: eteq_test_o2j
eteq_test_o2j: models/test/eteq.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/eteq.onnx --write /tmp/eteq.json
	mv /tmp/eteq.json models/test
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/eteq_ctx.onnx --write /tmp/eteq_ctx.json
	mv /tmp/eteq_ctx.json models/test

.PHONY: serial_test_o2j
serial_test_o2j: models/test/serial.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/serial.onnx --write /tmp/serial.json
	mv /tmp/serial.json models/test

.PHONY: oxsvc_test_o2j
oxsvc_test_o2j: models/test/local_oxsvc.onnx models/test/remote_oxsvc.onnx models/test/rootexpose_oxsvc.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/local_oxsvc.onnx --write /tmp/local_oxsvc.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/remote_oxsvc.onnx --write /tmp/remote_oxsvc.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/rootexpose_oxsvc.onnx --write /tmp/rootexpose_oxsvc.json
	mv /tmp/local_oxsvc.json models/test
	mv /tmp/remote_oxsvc.json models/test
	mv /tmp/rootexpose_oxsvc.json models/test

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
json2onnx: onnx_test_j2o eteq_test_j2o serial_test_j2o oxsvc_test_j2o gd_model_j2o dqn_model_j2o rbm_model_j2o dbn_model_j2o rnn_model_j2o lstm_model_j2o gru_model_j2o

.PHONY: onnx_test_j2o
onnx_test_j2o: models/test/bad_onnx.json models/test/bad_onnx2.json models/test/simple_onnx.json models/test/simple_stop.json models/test/layer_onnx.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/bad_onnx.json --write /tmp/bad_onnx.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/bad_onnx2.json --write /tmp/bad_onnx2.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/simple_onnx.json --write /tmp/simple_onnx.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/simple_stop.json --write /tmp/simple_stop.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/layer_onnx.json --write /tmp/layer_onnx.onnx
	mv /tmp/bad_onnx.onnx models/test
	mv /tmp/bad_onnx2.onnx models/test
	mv /tmp/simple_onnx.onnx models/test
	mv /tmp/simple_stop.onnx models/test
	mv /tmp/layer_onnx.onnx models/test

.PHONY: eteq_test_j2o
eteq_test_j2o: models/test/eteq.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/eteq.json --write /tmp/eteq.onnx
	mv /tmp/eteq.onnx models/test
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/eteq_ctx.json --write /tmp/eteq_ctx.onnx
	mv /tmp/eteq_ctx.onnx models/test

.PHONY: serial_test_j2o
serial_test_j2o: models/test/serial.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/serial.json --write /tmp/serial.onnx
	mv /tmp/serial.onnx models/test

.PHONY: oxsvc_test_j2o
oxsvc_test_j2o: models/test/local_oxsvc.json models/test/remote_oxsvc.json models/test/rootexpose_oxsvc.json
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/local_oxsvc.json --write /tmp/local_oxsvc.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/remote_oxsvc.json --write /tmp/remote_oxsvc.onnx
	bazel run //internal/onnx:inspector -- --read ${CURDIR}/models/test/rootexpose_oxsvc.json --write /tmp/rootexpose_oxsvc.onnx
	mv /tmp/local_oxsvc.onnx models/test
	mv /tmp/remote_oxsvc.onnx models/test
	mv /tmp/rootexpose_oxsvc.onnx models/test

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

TMP_COVFILE := /tmp/coverage.info
COVERAGE_INFO_FILE := bazel-out/_coverage/_coverage_report.dat
CCOVER := bazel coverage --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" --config gtest --config cc_coverage
COVERAGE_CTX := tmp/tenncor_coverage
CONVERSION_CSV := tmp/tenncor_conversions.csv
CTENNCOR_TESTS = $(shell bazel query //tenncor/... | grep test | grep -v -E 'srcs|//tenncor:ptest')

.PHONY: cov_clean
cov_clean:
	rm *.info
	rm -Rf html

# coverage helper
.PHONY: cov_init
cov_init:
	rm -Rf tmp
	mkdir -p $(COVERAGE_CTX)
	find . -maxdepth 1 | grep -E -v 'tmp|\.git|bazel-' | tail -n +2 | xargs -i cp -r {} $(COVERAGE_CTX)
	find $(COVERAGE_CTX) | grep -E '\.cpp|\.hpp' | python3 scripts/label_uniquify.py $(COVERAGE_CTX) > $(CONVERSION_CSV)
	find $(COVERAGE_CTX) | grep -E '\.yml' | python3 scripts/yaml_replace.py $(CONVERSION_CSV)

.PHONY: cov_copyout
cov_copyout:
	python3 scripts/label_replace.py $(COVERAGE_CTX)/$(COVFILE) $(CONVERSION_CSV) > $(COVFILE)

.PHONY: cov_genhtml
cov_genhtml: cov_copyout
	genhtml -o html $(COVFILE)

.PHONY: clean_test_coverage
clean_test_coverage: ${COVERAGE_INFO_FILE}
	lcov --remove ${COVERAGE_INFO_FILE} '**/test/*' '**/mock/*' '**/*.pb.*' -o ${TMP_COVFILE}

.PHONY: coverage
coverage:
	${CCOVER} //internal/... $(CTENNCOR_TESTS)
	@make clean_test_coverage
	lcov --extract ${TMP_COVFILE} 'internal/*' 'tenncor/*' -o coverage.info

###### INDIVIDUAL COVERAGES ######

.PHONY: cover_modules
cover_modules: cover_global cover_marsh cover_teq cover_eigen cover_onnx cover_query cover_opt cover_utils cover_distr cover_eteq cover_hone cover_layr cover_serial
	lcov \
		-a global_coverage.info \
		-a marsh_coverage.info \
		-a teq_coverage.info \
		-a eigen_coverage.info \
		-a onnx_coverage.info \
		-a query_coverage.info \
		-a opt_coverage.info \
		-a utils_coverage.info \
		-a distr_coverage.info \
		-a eteq_coverage.info \
		-a hone_coverage.info \
		-a layr_coverage.info \
		-a serial_coverage.info \
		-o modules_coverage.info

#### internal coverages ####

.PHONY: cover_global
cover_global:
	${CCOVER} --instrumentation_filter 'internal/global/*' //internal/global:test
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o global_coverage.info

.PHONY: cover_marsh
cover_marsh:
	${CCOVER} --instrumentation_filter 'internal/marsh/*' //internal/marsh/...
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o marsh_coverage.info

.PHONY: cover_teq
cover_teq:
	${CCOVER} --instrumentation_filter 'internal/teq/*' //internal/teq/...
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o teq_coverage.info

.PHONY: cover_eigen
cover_eigen:
	${CCOVER} --instrumentation_filter 'internal/eigen/*' //internal/eigen/...
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o eigen_coverage.info

.PHONY: cover_onnx
cover_onnx:
	${CCOVER} --instrumentation_filter 'internal/onnx/*' //internal/onnx/...
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o onnx_coverage.info

.PHONY: cover_query
cover_query:
	${CCOVER} --instrumentation_filter 'internal/query/*' //internal/query/...
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o query_coverage.info

.PHONY: cover_opt
cover_opt:
	${CCOVER} --instrumentation_filter 'internal/opt/*' //internal/opt/...
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o opt_coverage.info

.PHONY: cover_utils
cover_utils:
	${CCOVER} --instrumentation_filter 'internal/utils/*' //internal/utils/...
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o utils_coverage.info

#### tenncor coverages ####

.PHONY: cover_distr
cover_distr:
	${CCOVER} --instrumentation_filter 'tenncor/distr/*' //tenncor/distr/...
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o distr_coverage.info

.PHONY: cover_eteq
cover_eteq:
	${CCOVER} --instrumentation_filter 'tenncor/eteq/*' //tenncor/eteq/...
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o eteq_coverage.info

.PHONY: cover_hone
cover_hone:
	${CCOVER} --instrumentation_filter 'tenncor/hone/*' //tenncor/hone/...
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o hone_coverage.info

.PHONY: cover_layr
cover_layr:
	${CCOVER} --instrumentation_filter 'tenncor/layr/*' //tenncor/layr/...
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o layr_coverage.info

.PHONY: cover_serial
cover_serial:
	${CCOVER} --instrumentation_filter 'tenncor/serial/*' //tenncor/serial/...
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o serial_coverage.info

#### integration coverages ####

.PHONY: cover_integration
cover_integration:
	${CCOVER} --instrumentation_filter 'internal/*' 'tenncor/*' //tenncor:ctest
	@make clean_test_coverage
	lcov -a ${TMP_COVFILE} -o integ_coverage.info

.PHONY: conan_remote
conan_remote:
	conan remote add inexorgame "https://api.bintray.com/conan/inexorgame/inexor-conan"
	conan remote add mingkaic-co "https://gitlab.com/api/v4/projects/23299689/packages/conan"

build/conanbuildinfo.cmake:
	conan install -if build .

.PHONY: conan_install
conan_install: build/conanbuildinfo.cmake

.PHONY: conan_build
conan_build: build/conanbuildinfo.cmake
	conan build -bf build .

.PHONY: conan_create
conan_create:
	conan create . mingkaic-co/stable

.PHONY: conan_upload
conan_upload:
	conan upload tenncor/${VERSION}@mingkaic-co/stable --all --remote mingkaic-co

.PHONY: python_create
python_create:
	python3 setup.py sdist bdist_wheel

#### demo tryout ####

CC := gcc

.PHONY: try_demos
try_demos:
	bazel run --config ${CC}_eigen_optimal //demo:w2v
	bazel run --config ${CC}_eigen_optimal //demo:rnn -- --save /tmp/rnn.onnx
	bazel run --config ${CC}_eigen_optimal //demo:rbm -- --save /tmp/rbm.onnx
	bazel run --config ${CC}_eigen_optimal //demo:lstm_latin -- --save /tmp/latin_lstm.onnx
	bazel run --config ${CC}_eigen_optimal //demo:lstm_fast -- --save /tmp/fast_lstm.onnx
	bazel run --config ${CC}_eigen_optimal //demo:gru_latin -- --save /tmp/latin_gru.onnx
	bazel run --config ${CC}_eigen_optimal //demo:gru_fast -- --save /tmp/fast_gru.onnx
	bazel run --config ${CC}_eigen_optimal //demo:gd -- --save /tmp/gd.onnx
	bazel run --config ${CC}_eigen_optimal //demo:dqn -- --save /tmp/dqn.onnx
	bazel run --config ${CC}_eigen_optimal //demo:dbn -- --save /tmp/dbn.onnx
	bazel run --config ${CC}_eigen_optimal //demo:cgd
