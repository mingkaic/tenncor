#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
COV_DIR="$THIS_DIR";

lcov --base-directory . --directory . --zerocounters;
set -e

echo "===== ENVIRONMENT =====";
free -m;

# ===== Run Gtest =====
echo "===== TESTS =====";

source "$THIS_DIR/coverage.sh";

if [ "$1" == "fast" ]; then
	bzl_coverage //pbm:test //tag:test //teq:test //perf:test //eigen:test;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //gen:ptest
else
	bzl_coverage //ccur:test //eteq:ctest //layr:test //opt/...;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //eteq:ptest
fi

lcov --remove "$COV_DIR/coverage.info" 'external/*' '**/test/*' \
'testutil/*' '**/genfiles/*' 'dbg/*' -o "$COV_DIR/coverage.info";
send2codecov "$COV_DIR/coverage.info";

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
