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

Set RunTest option to ON during cmake generation should generate a default `libtenncor-inst.a` in the `bin` directory

	cmake -DTENNCOR_TEST=ON <path/to/tenncor>

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
		random_uniform<double> rinit(-1, 1);
		session& sess = session::get_instance();
	
		// initializes a 5 by 5 matrix with uniformly distributed
		// doubles between -1 and 1
		varptr<double> A = new variable<double>(common, rinit, "a");
		placeptr<double> B = new placeholder<double>(common, "b");
		varptr<double> C = matmul<double>::build(A, B);
		varptr<double> D = sigmoid<double>(C);
		
		sess.initialize_all<double>();
		B = std::vector<double>{...};
		
		gradient<double> grad(D);
		// prevent changes to B from cascading to gradient value
		grad.freeze();
		grad.execute();
		
		// forward accumulation
		tensor<double>* result = D->get_eval();
		// reverse accumulation
		tensor<double>* grad_result;
		grad.collect_grad(
		[&grad_result](inode<double>* key, 
					   placeholder<double>* value)
		{
			grad_result = value->get_eval();
		});
		
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
