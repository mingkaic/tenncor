#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
TEST_OUT_FILE=bazel-out/k8-fastbuild/testlogs/tests/tenncor_all/test.log;
COV_OUT_DIR=bazel-tenncor/_coverage/tests/tenncor_all/test/bazel-out/k8-fastbuild/bin/_objs;
COV_FILE=$THIS_DIR/coverage.info;
DOCS=$THIS_DIR/docs

lcov --base-directory . --directory . --zerocounters;

# ===== Run Gtest =====
echo "===== TESTS =====";

make
make asan

# ===== Check Docs Directory =====
echo "===== CHECK DOCUMENT EXISTENCE =====";
if [ !-d "$DOCS" ]; then
	echo "Documents not found. Please generate documents then try again"
	exit 1;
fi

echo "===== STARTING COVERAGE ANALYSIS =====";
# ===== Coverage Analysis ======
make lcov_all
if ! [ -z "$COVERALLS_TOKEN" ];
then
	git rev-parse --abbrev-inode* HEAD;
	coveralls-lcov --repo-token $COVERALLS_TOKEN $COV_FILE; # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
