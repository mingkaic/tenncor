#!/usr/bin/env bash

FUZZLOG=fuzz.out;
TIMEOUT=720; # 12 minute limit

lcov --base-directory . --directory . --zerocounters

# ===== Define Functions =====

assert_cmd() {
	eval timeout -s SIGKILL $TIMEOUT $*
	if [ $? -ne 0 ]; then
		echo "Command $* failed"
		cat $FUZZLOG
		exit 1;
	fi
	return $!
}

# ===== Run Gtest =====

# valgrind check (15 times)
for _ in {1..3}
do
    assert_cmd "bazel test --run_under=valgrind --test_output=all //..."
done

# regular checks (45 times)
for _ in {1..9}
do
    assert_cmd "bazel test --collect_code_coverage --test_output=all //..."
done

cat fuzz.out

# ===== Coverage Analysis ======
lcov --version
gcov --version
lcov --base-directory . --directory . --gcov-tool gcov-6 --capture --output-file coverage.info # capture coverage info
# filter out system and test code
lcov --remove coverage.info '**/gtest*' '**/tests/*' '/usr/*' --output-file coverage.info
lcov --list coverage.info # debug < see coverage here

if ! [ -z "$COVERALLS_TOKEN" ];
then
	git rev-parse --abbrev-ref HEAD;
	coveralls-lcov --repo-token ${COVERALLS_TOKEN} coverage.info # uploads to coveralls
fi

echo "";
echo "============ TENNCOR TEST SUCCESSFUL============";
