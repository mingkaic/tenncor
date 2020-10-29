#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
COV_DIR="$THIS_DIR";

CONTEXT=$(cd "$1" && pwd);

if (( $# > 1 )); then
	MODE="$2";
else
	MODE="all";
fi

if (( $# > 2 )); then
	COVMODE="$3";
else
	COVMODE="all";
fi

COVERAGE_CTX="$CONTEXT/tmp/tenncor_coverage";
CONVERSION_CSV="$CONTEXT/tmp/tenncor_conversion.csv";
TMP_COVFILE="$COV_DIR/coverage.info";
OUT_COVFILE="$COV_DIR/labelled_coverage.info";

rm -Rf "$COVERAGE_CTX";
mkdir -p "$COVERAGE_CTX";
find $CONTEXT -maxdepth 1 | grep -E -v 'tmp|\.git|bazel-' | tail -n +2 | xargs -i cp -r {} $COVERAGE_CTX;
find $COVERAGE_CTX | grep -E '\.cpp|\.hpp' | python3 "$THIS_DIR/label_uniquify.py" $COVERAGE_CTX > $CONVERSION_CSV;
find $COVERAGE_CTX | grep -E '\.yml' | python3 "$THIS_DIR/yaml_replace.py" $CONVERSION_CSV;

cd "$COVERAGE_CTX";
lcov --base-directory . --directory . --zerocounters;
set -e

echo "===== ENVIRONMENT =====";
if [ -x "$(command -v free)" ]; then
	free -m;
fi

# ===== Run Gtest =====
echo "===== TESTS =====";

source "$THIS_DIR/coverage.sh";

echo "Test Mode: $MODE";
echo "Coverage Mode: $COVMODE";
if [[ "$MODE" == "fast" ]]; then
	if [[ "$COVMODE" == "testonly" || "$COVMODE" == "all" ]]; then
		bzl_fulltest //internal/... $(bazel query //tenncor/... | grep test | grep -v -E 'srcs|//tenncor:ptest|//tenncor:ctest');
		bazel test --run_under='valgrind --leak-check=full' --remote_http_cache="$REMOTE_CACHE" //tools/...;
	fi

	if [[ "$COVMODE" == "coverage" || "$COVMODE" == "all" ]]; then
		bzl_coverage //internal/... $(bazel query //tenncor/... | grep test | grep -v -E 'srcs|//tenncor:ptest|//tenncor:ctest');
	fi
elif [[ "$MODE" == "integration" ]]; then
	if [[ "$COVMODE" == "testonly" || "$COVMODE" == "all" ]]; then
		bzl_fulltest //tenncor:ctest;
		bazel test --run_under='valgrind --leak-check=full' --remote_http_cache="$REMOTE_CACHE" //tenncor:ptest;
	fi

	if [[ "$COVMODE" == "coverage" || "$COVMODE" == "all" ]]; then
		bzl_coverage //tenncor:ctest;
	fi
else # test all
	if [[ "$COVMODE" == "testonly" || "$COVMODE" == "all" ]]; then
		bzl_fulltest //internal/... $(bazel query //tenncor/... | grep test | grep -v -E 'srcs|//tenncor:ptest');
		bazel test --run_under='valgrind --leak-check=full' --remote_http_cache="$REMOTE_CACHE" //tools/... //tenncor:ptest;
	fi

	if [[ "$COVMODE" == "coverage" || "$COVMODE" == "all" ]]; then
		bzl_coverage //internal/... $(bazel query //tenncor/... | grep test | grep -v -E 'srcs|//tenncor:ptest');
	fi
fi

if [[ "$COVMODE" == "coverage" || "$COVMODE" == "all" ]]; then
	python3 "$THIS_DIR/label_replace.py" $TMP_COVFILE $CONVERSION_CSV > $OUT_COVFILE;
	send2codecov "$COV_DIR/labelled_coverage.info";
fi
cd "$CONTEXT";

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
