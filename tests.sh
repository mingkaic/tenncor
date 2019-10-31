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

make lcov | grep -v '+' | grep -v 'Processing'

bazel test --run_under='valgrind --leak-check=full' //gen:ptest //eteq:ptest

bazel test --config asan --config gtest --action_env="ASAN_OPTIONS=detect_leaks=0" //perf:test

# ===== Coverage Analysis ======
if ! [ -z "$COVERALLS_TOKEN" ];
then
	echo "===== SENDING COVERAGE TO COVERALLS =====";
	git rev-parse --abbrev-inode* HEAD;
	coveralls-lcov --repo-token $COVERALLS_TOKEN $COV_FILE; # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
