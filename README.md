# Tenncor
[![Build Status](https://travis-ci.org/mingkaic/tenncor.svg?branch=master)](https://travis-ci.org/mingkaic/tenncor)
[![Coverage Status](https://coveralls.io/repos/github/mingkaic/tenncor/badge.svg)](https://coveralls.io/github/mingkaic/tenncor)

## Components

- ADE (Automatic Differentiation Engine)

This module supplies syntax tree for equation and generates derivative.
Constraints to the equation is limited to each tensor's shape.

- LLO (Low Level Operators)

This module is a sample library of data operators mapped to the ADE opcodes.
Expect this module to split when I decide to depend on external libraries (like eigen).

- PBM (Protobuf Marshaller)

This module marshals llo-extended ade graph

## Synopsis

The Tenncor libraries help developers build and evaluate tensor equations and its derivatives in C++.
A tensor is an N-dimensional numerical value container that organizes its content by some shape. An M by N matrix for instance, is a 2-dimensional tensor with a shape of <N, M> (in Tenncor).

## Building

Tenncor uses bazel 0.9+.

Download bazel: https://docs.bazel.build/versions/master/install.html
