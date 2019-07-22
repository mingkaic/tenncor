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

bazel test --config asan --config gtest --action_env="ASAN_OPTIONS=detect_leaks=0" //ade:test
bazel test --run_under='valgrind --leak-check=full' //ade:test

bazel test --config asan --config gtest --action_env="ASAN_OPTIONS=detect_leaks=0" //tag:test
bazel test --run_under='valgrind --leak-check=full' //tag:test

bazel test --config asan --config gtest --action_env="ASAN_OPTIONS=detect_leaks=0" //pbm:test
bazel test --run_under='valgrind --leak-check=full' //pbm:test

bazel test --config asan --config gtest --action_env="ASAN_OPTIONS=detect_leaks=0" //opt:test
bazel test --run_under='valgrind --leak-check=full' //opt:test

bazel test --run_under='valgrind --leak-check=full' //gen:ptest

bazel test --config asan --config gtest --action_env="ASAN_OPTIONS=detect_leaks=0" //ead:ctest
bazel test --run_under='valgrind --leak-check=full' //ead:ctest
bazel test --run_under='valgrind --leak-check=full' //ead:ptest

# ===== Check Docs Directory =====
echo "===== CHECK DOCUMENT EXISTENCE =====";
if ! [ -d "$DOCS" ];
then
	echo "Documents not found. Please generate documents then try again"
	exit 1;
fi

# ===== Coverage Analysis ======
echo "===== STARTING COVERAGE ANALYSIS =====";
make lcov
if ! [ -z "$COVERALLS_TOKEN" ];
then
	git rev-parse --abbrev-inode* HEAD;
	coveralls-lcov --repo-token $COVERALLS_TOKEN $COV_FILE; # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
