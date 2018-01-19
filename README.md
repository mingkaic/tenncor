# Tenncor
[![Build Status](https://travis-ci.org/mingkaic/tenncor.svg?branch=master)](https://travis-ci.org/mingkaic/tenncor)
[![Coverage Status](https://coveralls.io/repos/github/mingkaic/tenncor/badge.svg)](https://coveralls.io/github/mingkaic/tenncor)

## Synopsis

Tenncor library abstracts away tensor functions. 
A tensor is an N-dimensional geometric object. 

These objects are often used in machine learning models.
Quite often, the such system will need some derivative of the function. 

Tenncor simplifies the process by applying automatic differentiation, 
which takes advantage of chain rule to efficiently compute the desired derivative 
at any junction in the function graph. 

Tenncor builds the graphs such that data propogates reactively, 
meaning updates on individual leaves will trigger updates to corresponding ancestors.

Tensor operators are overloaded for custom nodes, 
so the resulting graph follows built-in C++ precedence and associativity rules.

## Building

Tenncor uses bazel 0.9+. 

Download bazel: https://docs.bazel.build/versions/master/install.html

## Testing

> bazel test //tests:tenncor_all

other tests:
- //tests:tenncor_connector
- //tests:tenncor_leaf
- //tests:tenncor_nodes
- //tests:tenncor_memory
- //tests:tenncor_operation
- //tests:tenncor_tensor

## API Reference

Working in Progress (Using doxygen)

## Examples

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

## Interesting Quirks

For the first 3 dimensions, shape values follow cartesian coordinates (x, y, z). 
This convention means that for matrices, rows are the second dimension, and columns are the first dimension, 
contrary to row and column major order. 

This will play a major role in matrix operations. An ongoing effort is made towards parameterizing shape conventions.

No special structured tensors are supported. (Example: symmetric, Toeplitz, positive definite)
