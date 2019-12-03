# Tenncor
[![Build Status](https://travis-ci.org/mingkaic/tenncor.svg?branch=master)](https://travis-ci.org/mingkaic/tenncor)
[![codecov](https://codecov.io/gh/mingkaic/tenncor/branch/master/graph/badge.svg)](https://codecov.io/gh/mingkaic/tenncor)

## Synopsis

Tenncor libraries help developers build and evaluate tensor equations and its derivatives.
A tensor is an N-dimensional container that organizes its content by some shape. An M by N matrix for instance, is a 2-dimensional tensor with a shape of [N, M] (according to Tenncor's x-y-z-... coordinate notation).

## Core Components

### [TEQ (Tensor EQuations)](teq/README_TEQ.md)

This module supplies syntax tree for equation and generates derivative.
Constraints to the equation is limited to each tensor's shape.

Tensor objects acts as the function graph scaffolding. Tensor scaffolding has the following actors:
- Session/Traveler access graph scaffolding.
- External Optimizer manipulates graph scaffolding.
- GradBuilder generates more graph scaffolding.

### [EIGEN (EIGEN wrapper)](eigen/README_EIGEN.md)

This module wraps eigen operators using TEQ shape and coordinate arguments.

Eigen objects hold the real data and provides API to manipulate the data.

### [ETEQ (Eigen TEQ)](eteq/README_ETEQ.md)

This module is implements basic operations for Tenncor's TEQ Tensor objects generated through pybinder.

Additionally, ETEQ also defines data format and (de)serialization methods required by PBM.

### [OPT (OPTimizer)](opt/README_OPT.md)

This module specifies graph optimization through TEQ's visitor pattern.

### [PBM (ProtoBuf Marshaller)](pbm/README_PBM.md)

This module marshals any TEQ graph, but requires data serialization functors when saving and loading.

## Supplemental Components

### [TAG (TAGger)](tag/README_TAG.md)

This module tags TEQ tensors with labels.

### [CCUR (ConCURrent session)](ccur/README_CCUR.md)

This module implements session that updates graph nodes concurrently

### [LAYR (LAYeR models)](layr/README_LAYR.md)

This module implements common machine learning models

### [DBG (Debug)](dbg/README_DBG.md)

This module is contains debug libraries for TEQ Graphs.

### General diagram

High-level diagram available: https://drive.google.com/file/d/1PrsFa7Duj4Whlu_m0lmFr5JGikGnU3gC/view?usp=sharing

## Tools and utility

- DBG (DeBuGger)

## Building

Tenncor uses bazel 0.28+.

Download bazel: https://docs.bazel.build/versions/master/install.html
