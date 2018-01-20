#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
TIMEOUT=900; # 15 minute limit
COV_OUT_FILE=bazel-out/k8-fastbuild/testlogs/tests/tenncor_all/coverage.dat;
TEST_OUT_FILE=bazel-out/k8-fastbuild/testlogs/tests/tenncor_all/test.log;

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

bazel build //...

# ===== Run Gtest =====
echo "===== STARTING TESTS =====";
# valgrind check (5 times)
assert_cmd "bazel test --run_under=valgrind //tests:tenncor_all";

# regular checks (45 times)
for _ in {1..9}
do
	assert_cmd "bazel coverage --instrumentation_filter= //tests:tenncor_all";
done

echo "===== STARTING COVERAGE ANALYSIS =====";
# ===== Coverage Analysis ======
ls bazel-out/k8-fastbuild/testlogs/tests/tenncor_all
lcov --version
gcov --version
lcov --list $COV_OUT_FILE; # debug < see coverage here
if ! [ -z "$COVERALLS_TOKEN" ];
then
	git rev-parse --abbrev-ref HEAD;
	coveralls-lcov --repo-token $COVERALLS_TOKEN $COV_OUT_FILE; # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
