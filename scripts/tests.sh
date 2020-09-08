#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
COV_DIR="$THIS_DIR";

lcov --base-directory . --directory . --zerocounters;
set -e

echo "===== ENVIRONMENT =====";
if [ -x "$(command -v free)" ]; then
	free -m;
fi

# ===== Run Gtest =====
echo "===== TESTS =====";

source "$THIS_DIR/coverage.sh";

if (( $# > 0 )); then
	MODE="$1";
else
	MODE="all";
fi

echo "Test Mode: $MODE";
if [[ "$MODE" == "fast" ]]; then
	bzl_coverage //internal/eigen:test //marsh:test //internal/onnx:test \
	//internal/opt:test //internal/query:test //internal/teq:test //internal/utils/...;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //tools/gen:ptest;
elif [[ "$MODE" == "eteq" ]]; then
	bzl_coverage //tenncor/eteq:ctest;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //tenncor/eteq:ptest;
elif [[ "$MODE" == "layr" ]]; then
	bzl_coverage //tenncor/layr:ctest;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //tenncor/layr:ptest;
elif [[ "$MODE" == "distrib" ]]; then
	bzl_coverage //tenncor/distr:ctest;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //tenncor/distr:ptest;
else
	bzl_coverage //internal/eigen:test //tenncor/eteq:ctest //tenncor/distr:ctest \
	//tenncor/layr:ctest //internal/marsh:test //internal/onnx:test //internal/opt:test \
	//internal/query:test //internal/teq:test //internal/utils/...;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" \
	//tools/gen:ptest //tenncor/eteq:ptest //tenncor/layr:ptest //tenncor/distr:ptest;
fi

lcov --remove "$COV_DIR/coverage.info" 'external/*' '**/test/*' \
'testutil/*' '**/genfiles/*' 'dbg/*' 'dbg/**/*' 'utils/*' 'utils/**/*' \
'perf/*' 'perf/**/*' '**/mock/*' '**/*.pb.h' '**/*.pb.cc' -o "$COV_DIR/coverage.info";
send2codecov "$COV_DIR/coverage.info";

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
