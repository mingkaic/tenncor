#!/usr/bin/env bash
eval "export $(make $1 2>/dev/null | grep -m 1 COVERAGE_OUTPUT_FILE=.*/coverage\.dat | cut -c 3-)"
COVERAGE_INFO_FILE=$2

sed -i 's:/.*/execroot/com_github_mingkaic_tenncor/::g' $COVERAGE_OUTPUT_FILE
lcov --remove $COVERAGE_OUTPUT_FILE '**/ioutil/*' -o $COVERAGE_INFO_FILE
