# Tenncor
[![Build Status](https://travis-ci.org/mingkaic/tenncor.svg?branch=master)](https://travis-ci.org/mingkaic/tenncor)
[![Coverage Status](https://coveralls.io/repos/github/mingkaic/tenncor/badge.svg)](https://coveralls.io/github/mingkaic/tenncor)

## Synopsis

Tenncor libraries help developers build and evaluate tensor equations and its derivatives.
A tensor is an N-dimensional container that organizes its content by some shape. An M by N matrix for instance, is a 2-dimensional tensor with a shape of [N, M] (according to Tenncor's x-y-z-... coordinate notation).

## Components

- [ADE (Automatic Differentiation Engine)](ade/README_ADE.md)

This module supplies syntax tree for equation and generates derivative.
Constraints to the equation is limited to each tensor's shape.

- [LLO (Low Level Operators)](llo/README_LLO.md)

This module is a sample library of data operators mapped to the ADE opcodes.
Expect this module to split when I decide to depend on external libraries (like eigen).

- [PBM (Protobuf Marshaller)](pbm/README_PBM.md)

This module marshals llo-extended ade graph

## Tools and utility

- DBG (Debugger)

## Building

Tenncor uses bazel 0.15+.

Download bazel: https://docs.bazel.build/versions/master/install.html
