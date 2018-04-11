#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
TIMEOUT=900; # 15 minute limit
TEST_OUT_FILE=bazel-out/k8-fastbuild/testlogs/tests/tenncor_all/test.log;
COV_OUT_DIR=bazel-tenncor/_coverage/tests/tenncor_all/test/bazel-out/k8-fastbuild/bin/_objs;
COV_FILE=coverage.info;
DOCS=$THIS_DIR/docs

lcov --base-directory . --directory . --zerocounters;

# ===== Define Functions =====

assert_cmd() {
	eval timeout -s SIGKILL $TIMEOUT $*
	if [ $? -ne 0 ]; then
		echo "Command $* failed";
		cat $TEST_OUT_FILE;
		exit 1;
	fi
	return $!
}

# ===== Prebuilt =====
echo "===== BUILD EVERYTHING =====";
bazel build //:...

# ===== Run Gtest =====
echo "===== STARTING TESTS =====";

# valgrind check (5 times)
export GTEST_REPEAT=5
assert_cmd "make memcheck";

# run coverage for all the tests
export GTEST_REPEAT=50
assert_cmd "make coverage";

# accept test
export GTEST_REPEAT=1
assert_cmd "make test_regress";

# ===== Check Docs Directory =====
echo "===== CHECK DOCUMENT EXISTENCE =====";
if [ -d "$DOCS" ]; then
	exit 1;
fi

echo "===== STARTING COVERAGE ANALYSIS =====";
# ===== Coverage Analysis ======
lcov --directory $COV_OUT_DIR --gcov-tool gcov-6 --capture --output-file $COV_FILE;
lcov --remove $COV_FILE '*/bazel-tenncor/external/*' '*/bazel-tenncor/bazel-out/*' '/usr/include/*' -o $COV_FILE;
lcov --list $COV_FILE;
if ! [ -z "$COVERALLS_TOKEN" ];
then
	sed -i 's:bazel-tenncor/::g' $COV_FILE;
	git rev-parse --abbrev-inode* HEAD;
	coveralls-lcov --repo-token $COVERALLS_TOKEN $COV_FILE; # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
