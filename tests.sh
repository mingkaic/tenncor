#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
TIMEOUT=900; # 15 minute limit

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
    assert_cmd "bazel coverage --test_output=all //...";
done

# ===== Coverage Analysis ======
lcov --version
gcov --version
lcov --directory bazel-out --gcov-tool gcov-6 --capture --output-file coverage.info # capture coverage info
lcov --list coverage.info # debug < see coverage here

if ! [ -z "$COVERALLS_TOKEN" ];
then
	git rev-parse --abbrev-ref HEAD;
	coveralls-lcov --repo-token ${COVERALLS_TOKEN} coverage.info # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL============";
