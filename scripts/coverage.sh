#!/usr/bin/env bash

# accepts tests as arguments
function bzl_coverage() {
	# run test before to detect errors
	bazel test \
	--action_env="ASAN_OPTIONS=detect_leaks=0" \
	--config asan \
	--config gtest \
	--remote_http_cache="$REMOTE_CACHE" $@
	# filter out Processing and '+' lines to avoid extreme verbosity
	bazel coverage \
	--action_env="ASAN_OPTIONS=detect_leaks=0" \
	--config asan \
	--config cc_coverage \
	--remote_http_cache="$REMOTE_CACHE" $@ | grep -v '+' | grep -v 'Processing';
	# extract coverage from bazel cache
	COV_FILE="coverage.info";
	if ! [ -z "$COV_DIR" ];
    then
		COV_FILE="$COV_DIR/coverage.info";
	fi
	lcov --remove bazel-out/_coverage/_coverage_report.dat '**/test/*' '**/mock/*' '**/*.pb.*' -o "$COV_FILE";
}

# uploads coverage file specified by the first argument to coveralls
function send2coverall() {
	lcov --list $1;
	if ! [ -z "$COVERALLS_TOKEN" ];
	then
		echo "===== SENDING COVERAGE TO COVERALLS =====";
		git rev-parse --abbrev-inode* HEAD;
		coveralls-lcov --repo-token $COVERALLS_TOKEN $1;
	fi
}

# uploads coverage file specified by the first argument to codecov
function send2codecov() {
	if ! [ -z "$CODECOV_TOKEN" ];
	then
		echo "===== SENDING COVERAGE TO CODECOV =====";
		bash <(curl -s https://codecov.io/bash) -f $1;
	fi
}
