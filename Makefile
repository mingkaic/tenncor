GTEST_REPEAT := 50

COMMON_BZL_FLAGS := --test_output=all --cache_test_results=no

GTEST_FLAGS := --action_env="GTEST_SHUFFLE=1" --action_env="GTEST_BREAK_ON_FAILURE=1"

REP_BZL_FLAG := --action_env="GTEST_REPEAT=$(GTEST_REPEAT)"

VALCHECK_BZL_FLAG := --run_under="valgrind"

BUILD := bazel build

RUN := bazel run

TEST := bazel test $(COMMON_BZL_FLAGS)

GTEST := $(TEST) $(GTEST_FLAGS)

COVER := bazel coverage $(COMMON_BZL_FLAGS) $(GTEST_FLAGS)

# SERIALS := serial_cc serial_py

# GRAPHMGRS := graphmgr_cc graphmgr_py

# all: proto test_py proto_build unit_test test_regress

# travis_test: build test_py proto_build memcheck coverage test_regress

# build protobuf files
# build: tenncor_build proto

# tenncor_build:
# 	$(BUILD) //:tenncor

# proto: monitor $(SERIALS) $(GRAPHMGRS)

# monitor:
# 	$(BUILD) //proto:tenncor_monitor_grpc

# $(SERIALS):
# 	$(BUILD) //proto:tenncor_$@_proto

# $(GRAPHMGRS):
# 	$(BUILD) //tests/graphmgr:$@_grpc

# # python data build and test
# proto_build:
# 	$(RUN) //tests/py:protogen -- $(shell pwd)/tests/unit/samples

test: unittest regression

# unit test
unittest: test_clay test_mold test_slip test_kiln test_wire

test_clay:
	$(GTEST) $(REP_BZL_FLAG) //clay:test

test_mold:
	$(GTEST) $(REP_BZL_FLAG) //mold:test

test_slip:
	$(GTEST) $(REP_BZL_FLAG) //slip:test

test_kiln:
	$(GTEST) $(REP_BZL_FLAG) //kiln:test

test_wire:
	$(GTEST) $(REP_BZL_FLAG) //wire:test

# valgrind unit tests
valgrind: valg_clay valg_mold valg_slip valg_kiln valg_wire

valg_clay:
	$(GTEST) $(VALCHECK_BZL_FLAG) --action_env="GTEST_REPEAT=5" //clay:test

valg_mold:
	$(GTEST) $(VALCHECK_BZL_FLAG) --action_env="GTEST_REPEAT=5" //mold:test

valg_slip:
	$(GTEST) $(VALCHECK_BZL_FLAG) --action_env="GTEST_REPEAT=5" //slip:test

valg_kiln:
	$(GTEST) $(VALCHECK_BZL_FLAG) --action_env="GTEST_REPEAT=5" //kiln:test

valg_wire:
	$(GTEST) $(VALCHECK_BZL_FLAG) --action_env="GTEST_REPEAT=5" //wire:test

# asan unit tests

# coverage unit tests
coverage: cover_clay cover_mold cover_slip cover_kiln cover_wire

cover_clay:
	$(COVER) $(REP_BZL_FLAG) //clay:test --instrumentation_filter=/clay[/:]

cover_mold:
	$(COVER) $(REP_BZL_FLAG) //mold:test --instrumentation_filter=/mold[/:]

cover_slip:
	$(COVER) $(REP_BZL_FLAG) //slip:test --instrumentation_filter=/slip[/:]

cover_kiln:
	$(COVER) $(REP_BZL_FLAG) //kiln:test --instrumentation_filter=/kiln[/:]

cover_wire:
	$(COVER) $(REP_BZL_FLAG) //wire:test --instrumentation_filter=/wire[/:]

# regression testing
regression:
	$(GTEST) //regress:test

# todo: deprecate
acceptdata: cleandata
	python regress/tf_generate/tf_generate.py

# remove all test data
clean_data:
	rm -f tests/regress/samples/*
	rm -f tests/unit/samples/*

clean: clean_data
	bazel clean
