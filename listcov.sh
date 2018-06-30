#!/usr/bin/env bash
sed -i 's:/.*/execroot/com_github_mingkaic_tenncor/::g' $COVERAGE_OUTPUT_FILE
lcov --remove $COVERAGE_OUTPUT_FILE 'external/*' 'tests/*' -o coverage.info
lcov --list coverage.info
