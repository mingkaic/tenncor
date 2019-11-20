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
	bzl_coverage //pbm:test //tag:test //teq:test //perf:test //eigen:test //marsh:test;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //gen:ptest;
elif [[ "$MODE" == "slow" ]]; then
	bzl_coverage //ccur:test //eteq:ctest //layr:test //opt/...;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //eteq:ptest;
else
	bzl_coverage //pbm:test //tag:test //teq:test //perf:test //eigen:test //marsh:test
		//ccur:test //eteq:ctest //layr:test //opt/...;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //eteq:ptest //gen:ptest;
fi

lcov --remove "$COV_DIR/coverage.info" 'external/*' '**/test/*' \
'testutil/*' '**/genfiles/*' 'dbg/*' '**/mock/*' -o "$COV_DIR/coverage.info";
send2codecov "$COV_DIR/coverage.info";

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
