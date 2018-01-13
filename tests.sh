#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
TIMEOUT=900; # 15 minute limit
COVERAGE_OUTPUT_FILE=bazel-out/k8-fastbuild/testlogs/tests/tenncor_test/coverage.dat

lcov --base-directory . --directory . --zerocounters

# ===== Define Functions =====

assert_cmd() {
	eval timeout -s SIGKILL $TIMEOUT $*
	if [ $? -ne 0 ]; then
		echo "Command $* failed"
		exit 1;
	fi
	return $!
}

# ===== Run Gtest =====

bazel build //...

# valgrind check (5 times)
assert_cmd "bazel test --run_under=valgrind --test_output=all //...";

# regular checks (45 times)
for _ in {1..9}
do
	assert_cmd "bazel coverage --instrumentation_filter= --test_output=all //...";
done

# ===== Coverage Analysis ======
lcov --version
gcov --version
lcov --list $COVERAGE_OUTPUT_FILE # debug < see coverage here

if ! [ -z "$COVERALLS_TOKEN" ];
then
	git rev-parse --abbrev-ref HEAD;
	coveralls-lcov --repo-token $COVERALLS_TOKEN $COVERAGE_OUTPUT_FILE # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL============";
