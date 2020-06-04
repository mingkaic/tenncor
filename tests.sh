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
	bzl_coverage //eigen:test //marsh:test //onnx:test \
	//opt:test //perf:test //query:test //teq:test;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //gen:ptest;
elif [[ "$MODE" == "slow" ]]; then
	bzl_coverage //eteq:ctest //layr:ctest;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //eteq:ptest //layr:ptest;
else
	bzl_coverage //eigen:test //eteq:ctest //layr:ctest //marsh:test \
	//onnx:test //opt:test //perf:test //query:test //teq:test;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //eteq:ptest //gen:ptest //layr:ptest;
fi

lcov --remove "$COV_DIR/coverage.info" 'external/*' '**/test/*' \
'testutil/*' '**/genfiles/*' 'dbg/*' 'dbg/**/*' 'utils/*' 'utils/**/*' \
'perf/*' 'perf/**/*' '**/mock/*' '**/*.pb.h' '**/*.pb.cc' -o "$COV_DIR/coverage.info";
send2codecov "$COV_DIR/coverage.info";

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
