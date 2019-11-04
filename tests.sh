#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
COV_FILE=$THIS_DIR/coverage.info;
DOCS=$THIS_DIR/docs

lcov --base-directory . --directory . --zerocounters;
set -e

echo "===== ENVIRONMENT =====";
free -m;

# ===== Run Gtest =====
echo "===== TESTS =====";

bazel coverage --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" \
--config cc_coverage --remote_http_cache="$REMOTE_CACHE" //ccur:test \
//eteq:ctest //layr:test //opt/... //pbm:test //tag:test //teq:test \
| grep -v '+' | grep -v 'Processing'
lcov --remove bazel-out/_coverage/_coverage_report.dat -o coverage.info
lcov --remove coverage.info 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' 'dbg/*' -o coverage.info

bazel test --run_under='valgrind --leak-check=full' \
--remote_http_cache="$REMOTE_CACHE" //gen:ptest //eteq:ptest

bazel test --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" \
--remote_http_cache="$REMOTE_CACHE" //perf:test

# ===== Coverage Analysis ======
lcov --list coverage.info

if ! [ -z "$COVERALLS_TOKEN" ];
then
	echo "===== SENDING COVERAGE TO COVERALLS =====";
	git rev-parse --abbrev-inode* HEAD;
	coveralls-lcov --repo-token $COVERALLS_TOKEN $COV_FILE; # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
