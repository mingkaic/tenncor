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

bazel test --config asan --config gtest --action_env="ASAN_OPTIONS=detect_leaks=0" --define ETEQ_CFG=MIN \
//teq:test //tag:test //pbm:test //opt:test //opt/parse:test //eteq:ctest //perf:test //pll:test

bazel test --run_under='valgrind --leak-check=full' --define ETEQ_CFG=MIN \
//teq:test //gen:ptest //tag:test //pbm:test //opt:test //opt/parse:test //eteq:ctest //eteq:ptest //perf:test //pll:test

# ===== Coverage Analysis ======
echo "===== STARTING COVERAGE ANALYSIS =====";
make lcov | grep -v '+' | grep -v 'Processing'

if ! [ -z "$COVERALLS_TOKEN" ];
then
	git rev-parse --abbrev-inode* HEAD;
	coveralls-lcov --repo-token $COVERALLS_TOKEN $COV_FILE; # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
