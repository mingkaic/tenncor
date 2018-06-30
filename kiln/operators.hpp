/*!
 *
 *  operators.hpp
 *  kiln
 *
 *  Purpose:
 *  helper functions to build kiln Functors
 *
 *  Created by Mingkai Chen on 2016-10-24.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "kiln/constant.hpp"
#include "kiln/functor.hpp"

#pragma once
#ifndef KILN_OPERATORS_HPP
#define KILN_OPERATORS_HPP

namespace kiln
{

using GradMapT = std::unordered_map<slip::OPCODE,GradF,slip::EnumHash>;

extern const GradMapT grad_op;

//! convert a to match type
Identifier* cast (Identifier* type, Identifier* a);

//! absolute value of a
Identifier* abs (Identifier* a);

//! negative value of a
Identifier* neg (Identifier* a);

//! operator ! of a
Identifier* logical_not (Identifier* a);

//! sin of a
Identifier* sin (Identifier* a);

//! cos of a
Identifier* cos (Identifier* a);

//! tan of a
Identifier* tan (Identifier* a);

//! e of power a
Identifier* exp (Identifier* a);

//! natural log a
Identifier* log (Identifier* a);

//! square root of a
Identifier* sqrt (Identifier* a);

//! round a
Identifier* round (Identifier* a);

//! transpose, default perm is same as behavior n-1 ... 0
Identifier* transpose (Identifier* a);

Identifier* transpose (Identifier* a, Identifier* perm);

Identifier* transpose (Identifier* a, std::vector<uint64_t> perm);

//! flip a in specified dimensions
Identifier* flip (Identifier* a, Identifier* dims);

Identifier* flip (Identifier* a, std::vector<uint64_t> dims);

//! repeat a n times along inserted dimension dim
Identifier* expand (Identifier* a, Identifier* n, Identifier* dim);

Identifier* expand (Identifier* a, Identifier* n, uint64_t dim);

Identifier* expand (Identifier* a, uint64_t n, uint64_t dim);

//! get the number of elements in a
Identifier* n_elems (Identifier* a);

//! get the number of elements in across a dimension in a
Identifier* n_dimension (Identifier* a, Identifier* dim);

Identifier* n_dimension (Identifier* a, uint64_t dim);

//! dimensioned operators
//! b to the power of x
Identifier* pow (Identifier* b, Identifier* x);

//! add a and b
Identifier* add (Identifier* a, Identifier* b);

//! subtract a and b
Identifier* sub (Identifier* a, Identifier* b);

//! multiply a and b
Identifier* mul (Identifier* a, Identifier* b);

//! divide a and b
Identifier* div (Identifier* a, Identifier* b);

//! a == b
Identifier* eq (Identifier* a, Identifier* b);

//! a != b
Identifier* neq (Identifier* a, Identifier* b);

//! a < b
Identifier* lt (Identifier* a, Identifier* b);

//! a > b
Identifier* gt (Identifier* a, Identifier* b);

Identifier* pow (Identifier* b, mold::Range xdim,
	Identifier* x, mold::Range bdim);

Identifier* add (Identifier* a, mold::Range adim,
	Identifier* b, mold::Range bdim);

Identifier* sub (Identifier* a, mold::Range adim,
	Identifier* b, mold::Range bdim);

Identifier* mul (Identifier* a, mold::Range adim,
	Identifier* b, mold::Range bdim);

Identifier* div (Identifier* a, mold::Range adim,
	Identifier* b, mold::Range bdim);

Identifier* eq (Identifier* a, mold::Range adim,
	Identifier* b, mold::Range bdim);

Identifier* neq (Identifier* a, mold::Range adim,
	Identifier* b, mold::Range bdim);

Identifier* lt (Identifier* a, mold::Range adim,
	Identifier* b, mold::Range bdim);

Identifier* gt (Identifier* a, mold::Range adim,
	Identifier* b, mold::Range bdim);

//! generate data of within binomial distribution given (n, p)
Identifier* binomial_sample (Identifier* n, Identifier* p);

Identifier* binomial_sample (Identifier* n, double p);

//! generate data of within uniform distribution given (min, max)
Identifier* uniform_sample (Identifier* min, Identifier* max);

//! generate data of within normal distribution given (mean, stdev)
Identifier* normal_sample (Identifier* mean, Identifier* stdev);

//! obtain the index of max value in a, lack of or invalid dimension look across all of a
Identifier* arg_max (Identifier* a);

Identifier* arg_max (Identifier* a, Identifier* dim);

Identifier* arg_max (Identifier* a, uint64_t dim);

//! obtain the max of a, lack of or invalid dimension look across all of a
Identifier* reduce_max (Identifier* a);

Identifier* reduce_max (Identifier* a, Identifier* dim);

Identifier* reduce_max (Identifier* a, uint64_t dim);

//! obtain the sum of a, lack of or invalid dimension look across all of a
Identifier* reduce_sum (Identifier* a);

Identifier* reduce_sum (Identifier* a, Identifier* dim);

Identifier* reduce_sum (Identifier* a, uint64_t dim);

//! matrix multiplication (todo: expand to include matmul along other dimensions, currently {0, 1} only)
Identifier* matmul (Identifier* a, Identifier* b);


//! obtain the mean of a, lack of or invalid dimension look across all of a
Identifier* reduce_mean (Identifier* a);

Identifier* reduce_mean (Identifier* a, Identifier* dim);

Identifier* reduce_mean (Identifier* a, uint64_t dim);

//! obtain the l2norm of a, lack of or invalid dimension look across all of a
Identifier* reduce_l2norm (Identifier* a);

Identifier* reduce_l2norm (Identifier* a, Identifier* dim);

Identifier* reduce_l2norm (Identifier* a, uint64_t dim);

//! clip values in range [min, max]
Identifier* clip (Identifier* a, Identifier* min, Identifier* max);

//! normalize clip values with capacity cap
Identifier* clip_norm (Identifier* a, Identifier* cap);

Identifier* reshape (Identifier* a, Identifier* shape);

Identifier* reshape (Identifier* a, std::vector<uint64_t> shape);

Identifier* jacobian (Identifier* a, Identifier* b, Identifier* dims);

Identifier* jacobian (Identifier* a, Identifier* b, uint64_t targetdim, uint64_t swapdim);

Identifier* trace_expand (Identifier* a, Identifier* dim);

Identifier* trace_expand (Identifier* a, uint64_t dim);

// unimplemented

//! for example: window {0, 1} gives output f[i, j, :] = sum(a[i:i+filtshape[0], j:j+filtshape[1], :] * filter)
//! whereas window {0,2} gives output f[i, :, j] = sum(a[i:i+filtshape[0], :, j:j+filtshape[1]] * filter)
Identifier* cross_corr2d (Identifier* a, Identifier* filter, std::pair<uint64_t,uint64_t> dims = {0, 1});

//! convolve a with filter, conv(a, filter, dims) = cross_conv(a, flip(filter), dims)
Identifier* conv2d (Identifier* a, Identifier* filter, std::pair<uint64_t,uint64_t> dims = {0, 1});

}

#endif /* TENNCOR_OPERATIONS_HPP */
