#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
TIMEOUT=900; # 15 minute limit
COV_OUT_DIR=bazel-out/k8-fastbuild/testlogs/tests/tenncor_

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
	assert_cmd "bazel coverage --instrumentation_filter=//:tenncor --test_output=all //...";
done

# ===== Coverage Analysis ======
lcov --version
gcov --version
lcov -a ${COV_OUT_DIR}connector/coverage.dat -a ${COV_OUT_DIR}leaf/coverage.dat \
	-a ${COV_OUT_DIR}memory/coverage.dat -a ${COV_OUT_DIR}nodes/coverage.dat \
	-a ${COV_OUT_DIR}operation/coverage.dat -a ${COV_OUT_DIR}tensor/coverage.dat \
	-o coverage.info

lcov --list coverage.info # debug < see coverage here

if ! [ -z "$COVERALLS_TOKEN" ];
then
	git rev-parse --abbrev-ref HEAD;
	coveralls-lcov --repo-token $COVERALLS_TOKEN coverage.info # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL============";
