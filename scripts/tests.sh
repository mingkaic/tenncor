#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";
COV_DIR="$THIS_DIR";

CONTEXT=$(cd "$1" && pwd);

if (( $# > 1 )); then
	MODE="$2";
else
	MODE="all";
fi

WORKDIR="$CONTEXT/tmp/tenncor_coverage";
CONVERSION_CSV="$CONTEXT/tmp/tenncor_conversion.csv";

rm -Rf "$WORKDIR";
mkdir -p "$WORKDIR";
find $WORKDIR -maxdepth 1 | grep -E -v 'tmp|.git|bazel-' | tail -n +2 | xargs -i cp -r {} $WORKDIR;
find $WORKDIR | grep -E '.cpp|.hpp' | python3 scripts/label_uniquify.py $WORKDIR > $CONTEXT;
find $WORKDIR | grep -E '.yml' | python3 scripts/yaml_replace.py $CONTEXT;

cd "$WORKDIR";
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
if [[ "$MODE" == "fast" ]]; then
	bzl_coverage //tenncor/... $(bazel query //tenncor/... | grep test | grep -v -E 'srcs|//tenncor:ptest|//tenncor:ctest');

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //tools/...;
elif [[ "$MODE" == "integration" ]]; then
	bzl_coverage //tenncor:ctest;

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //tenncor:ptest;
else # test all
	bzl_coverage //internal/... $(bazel query //tenncor/... | grep test | grep -v -E 'srcs|//tenncor:ptest');

	bazel test --run_under='valgrind --leak-check=full' \
	--remote_http_cache="$REMOTE_CACHE" //tools/... //tenncor:ptest;
fi

python3 "$THIS_DIR/label_replace.py" "$COV_DIR/coverage.info" $CONVERSION_CSV > "$COV_DIR/labelled_coverage.info";
send2codecov "$COV_DIR/labelled_coverage.info";
cd "$CONTEXT";

echo "";
echo "============ TENNCOR TEST SUCCESSFUL ============";
