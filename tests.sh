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

bzl_coverage //ccur:test //eteq:ctest //layr:test //opt/... \
//perf:test //pbm:test //tag:test //teq:test;

bazel test --run_under='valgrind --leak-check=full' \
--remote_http_cache="$REMOTE_CACHE" //gen:ptest //eteq:ptest;

lcov --remove "$COV_DIR/coverage.info" 'external/*' '**/test/*' \
'testutil/*' '**/genfiles/*' 'dbg/*' -o "$COV_DIR/coverage.info";
send2coverall "$COV_DIR/coverage.info";

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
