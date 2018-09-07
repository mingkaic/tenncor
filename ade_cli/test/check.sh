#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

BIN=$1
INPUT_DIR=$THIS_DIR/inputs
EXPECT_DIR=$THIS_DIR/expects

for FILE in $INPUT_DIR/*; do
	FILENAME=${FILE##*/}
	if [ -f $EXPECT_DIR/$FILENAME ]; then
		echo "diffing $FILENAME"
		echo "================="
		diff $EXPECT_DIR/$FILENAME <($BIN $INPUT_DIR/$FILENAME);
	fi
done
