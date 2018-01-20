#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
TIMEOUT=900; # 15 minute limit
TEST_OUT_FILE=bazel-out/k8-fastbuild/testlogs/tests/tenncor_all/test.log;
COV_OUT_DIR=bazel-tenncor/_coverage/tests/tenncor_all/test/bazel-out/k8-fastbuild/bin/_objs;

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
lcov --directory $COV_OUT_DIR --gcov-tool gcov-6 --capture --output-file coverage.info;
lcov --remove coverage.info '*/bazel-tenncor/external/*' '*/bazel-tenncor/bazel-out/*' '/usr/include/*' -o coverage.info;
lcov --list coverage.info;
if ! [ -z "$COVERALLS_TOKEN" ];
then
	git rev-parse --abbrev-ref HEAD;
	coveralls-lcov --repo-token $COVERALLS_TOKEN overage.info; # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
