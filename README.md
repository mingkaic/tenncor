# Tenncor
[![Build Status](https://travis-ci.org/mingkaic/tenncor.svg?branch=master)](https://travis-ci.org/mingkaic/tenncor)
[![Coverage Status](https://coveralls.io/repos/github/mingkaic/tenncor/badge.svg?branch=master)](https://coveralls.io/github/mingkaic/tenncor?branch=master)

## Synopsis

Tenncor libraries help developers build and evaluate tensor equations and its derivatives.
A tensor is an N-dimensional container that organizes its content by some shape. An M by N matrix for instance, is a 2-dimensional tensor with a shape of [N, M] (according to Tenncor's x-y-z-... coordinate notation).

This project aims for modularity. Everything (including this project) should be easily replaceable in any high-level system.

## Components

- [ADE (Automatic Differentiation Engine)](ade/README_ADE.md)

This module supplies syntax tree for equation and generates derivative.
Constraints to the equation is limited to each tensor's shape.

- [AGE (ADE Generation Engine)](age/README_AGE.md)

This generator creates glue layer between ADE and data manipulation libraries as well as map operational codes to its respective chain rule.

- [BWD (Backward operations)](bwd/README_BWD.md)

This library provides traveler for generating partial derivative equations using some set of chain rules.

## Tools and utility

- DBG (Debugger)

## Building

Tenncor uses bazel 0.15+.

Download bazel: https://docs.bazel.build/versions/master/install.html
