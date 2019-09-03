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

bazel test --config asan --config gtest --action_env="ASAN_OPTIONS=detect_leaks=0" --define EAD_CFG=MIN \
//ade:test //tag:test //pbm:test //opt:test //opt/parse:test //ead:ctest //perf:test //pll:test

bazel test --run_under='valgrind --leak-check=full' --define EAD_CFG=MIN \
//ade:test //gen:ptest //tag:test //pbm:test //opt:test //opt/parse:test //ead:ctest //ead:ptest //perf:test //pll:test

# ===== Coverage Analysis ======
echo "===== STARTING COVERAGE ANALYSIS =====";
bazel coverage --config asan --action_env="ASAN_OPTIONS=detect_leaks=0" --define EAD_CFG=MIN \
--config gtest --config cc_coverage \
//ade:test //tag:test //pbm:test //opt:test //opt/parse:test //ead:ctest //perf:test //pll:test
lcov --remove bazel-out/_coverage/_coverage_report.dat \
'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' 'dbg/*' \
-o coverage.info
lcov --list coverage.info | grep -v '+' | grep -v 'Processing'

if ! [ -z "$COVERALLS_TOKEN" ];
then
	git rev-parse --abbrev-inode* HEAD;
	coveralls-lcov --repo-token $COVERALLS_TOKEN $COV_FILE; # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
