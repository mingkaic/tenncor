# Tenncor
[![Build Status](https://travis-ci.org/mingkaic/tenncor.svg?branch=master)](https://travis-ci.org/mingkaic/tenncor)
[![Coverage Status](https://coveralls.io/repos/github/mingkaic/tenncor/badge.svg)](https://coveralls.io/github/mingkaic/tenncor)

## Synopsis

The Tenncor library facilitates developers in coding mathematical tensor functions in C++.
Through Tenncor, developers can obtain the tensor function's nth derivative via automatic differentiation. A tensor is an N-dimensional numerical value container that organizes its content by some shape. A M by N matrix for instance, is a 2-dimensional tensor with a shape of <N, M> (in Tenncor).

Tensor abstractions are often useful in machine learning models because features (represented as a tensor) are almost always transformed in an ordered fashion. Since transforming arrays with respect to shape and type information is often tedious, cython frameworks are currently the best tools for ML.
Tenncor aims to provide cython's elegant interface in C++ with minimal dependenices.

## Building

Tenncor uses bazel 0.9+. 

Download bazel: https://docs.bazel.build/versions/master/install.html

## Testing

Run

> bazel test //tests:tenncor_all

other tests:
- //tests:tenncor_connector
- //tests:tenncor_leaf
- //tests:tenncor_nodes
- //tests:tenncor_memory
- //tests:tenncor_operation
- //tests:tenncor_tensor

## Monitoring

use libraries

- //:tenncor_csv
- //:tenncor_rpc

## Example

	#include "executor/gradient.hpp"
	#include "graph/varptr.hpp"
	#include "graph/leaf/variable.hpp"
	#include "graph/connector/immutable/matmul.hpp"
	
	using namespace nnet;
	
	int main () {
		tensorshape common = std::vector<size_t>{5, 5};
		random_uniform rinit(-1, 1);
	
		// initializes a 5 by 5 matrix with uniformly distributed
		// doubles between -1 and 1
		variable* A = new variable(common, rinit, tenncor::tensor_proto::DOUBLE_T, "a");
		placeptr B = new placeholder(common, tenncor::tensor_proto::DOUBLE_T, "b");
		varptr C = matmul(varptr(A), B);
		varptr D = sigmoid(C);
		
		A->initialize();
		B = std::vector<double>{...};
		
		varptr grad = D->derive(A);
		
		// forward accumulation
		itensor* result = D->get_eval();
		// reverse accumulation
		itensor* grad_result = grad->eval();

		std::vector<double> raw_data = expose<double>(D);
		std::vector<double> raw_grad = expose<double>(grad);
		
		delete A;
		delete B;
		
		return 0;
	} 

## Todos and Quirks

For the first 3 dimensions, shape values follow cartesian coordinates (x, y, z). This convention means that for matrices, rows are the second dimension, and columns are the first dimension, contrary to row and column major order. 

An ongoing effort is made towards parameterizing shape conventions.

No special structured tensors are supported. (Example: symmetric, Toeplitz, positive definite)

Tensor operation optimizations (minimizing intermediate values and instigating multithreading) are in progress.
