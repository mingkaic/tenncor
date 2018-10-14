#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

cd $THIS_DIR

rm -rf docs

doxygen
mv doxout/html docs

rm -rf doxout
