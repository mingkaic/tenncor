/*!
 *
 *  operators.hpp
 *  wire
 *
 *  Purpose:
 *  helper functions to build wire Functors
 *
 *  Created by Mingkai Chen on 2016-10-24.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/constant.hpp"
#include "include/graph/functor.hpp"

#pragma once
#ifndef WIRE_OPERATORS_HPP
#define WIRE_OPERATORS_HPP

namespace wire
{

void assert_type (Identifier* a, TENS_TYPE type);

void assert_shape (Identifier* a, tshape shape);

//! absolute value of a
Identifier* abs (Identifier* a);

//! negative value of a
Identifier* neg (Identifier* a);

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

//! b to the power of x
Identifier* pow (Identifier* b, Identifier* x);

template <typename T>
Identifier* pow (Identifier* b, T x) { return pow(b, Constant::get<T>(x)); }

template <typename T>
Identifier* pow (T b, Identifier* x) { return pow(Constant::get<T>(b), x); }

//! add a and b
Identifier* add (Identifier* a, Identifier* b);

template <typename T>
Identifier* add (Identifier* a, T b) { return add(a, Constant::get<T>(b)); }

template <typename T>
Identifier* add (T a, Identifier* b) { return add(Constant::get<T>(a), b); }

//! subtract a and b
Identifier* sub (Identifier* a, Identifier* b);

template <typename T>
Identifier* sub (Identifier* a, T b) { return sub(a, Constant::get<T>(b)); }

template <typename T>
Identifier* sub (T a, Identifier* b) { return sub(Constant::get<T>(a), b); }

//! multiply a and b
Identifier* mul (Identifier* a, Identifier* b);

template <typename T>
Identifier* mul (Identifier* a, T b) { return mul(a, Constant::get<T>(b)); }

template <typename T>
Identifier* mul (T a, Identifier* b) { return mul(Constant::get<T>(a), b); }

//! divide a and b
Identifier* div (Identifier* a, Identifier* b);

template <typename T>
Identifier* div (Identifier* a, T b) { return div(a, Constant::get<T>(b)); }

template <typename T>
Identifier* div (T a, Identifier* b) { return div(Constant::get<T>(a), b); }

//! a == b
Identifier* eq (Identifier* a, Identifier* b);

template <typename T>
Identifier* eq (Identifier* a, T b) { return eq(a, Constant::get<T>(b)); }

template <typename T>
Identifier* eq (T a, Identifier* b) { return eq(Constant::get<T>(a), b); }

//! a != b
Identifier* neq (Identifier* a, Identifier* b);

template <typename T>
Identifier* neq (Identifier* a, T b) { return neq(a, Constant::get<T>(b)); }

template <typename T>
Identifier* neq (T a, Identifier* b) { return neq(Constant::get<T>(a), b); }

//! a < b
Identifier* lt (Identifier* a, Identifier* b);

template <typename T>
Identifier* lt (Identifier* a, T b) { return lt(a, Constant::get<T>(b)); }

template <typename T>
Identifier* lt (T a, Identifier* b) { return lt(Constant::get<T>(a), b); }

//! a > b
Identifier* gt (Identifier* a, Identifier* b);

template <typename T>
Identifier* gt (Identifier* a, T b) { return gt(a, Constant::get<T>(b)); }

template <typename T>
Identifier* gt (T a, Identifier* b) { return gt(Constant::get<T>(a), b); }

//! generate data of within binomial distribution given (n, p)
Identifier* binomial_sample (Identifier* n, Identifier* p);

Identifier* binomial_sample (Identifier* n, double p);

template <typename T>
Identifier* binomial_sample (T n, Identifier* p)
{ return binomial_sample(Identifier*(Constant::get<T>(n)), p); }

//! generate data of within uniform distribution given (min, max)
Identifier* uniform_sample (Identifier* min, Identifier* max);

template <typename T>
Identifier* uniform_sample (Identifier* min, T max)
{ return uniform_sample(min, Identifier*(Constant::get<T>(max))); }

template <typename T>
Identifier* uniform_sample (T min, Identifier* max)
{ return uniform_sample(Identifier*(Constant::get<T>(min)), max); }

//! generate data of within normal distribution given (mean, stdev)
Identifier* normal_sample (Identifier* mean, Identifier* stdev);

template <typename T>
Identifier* normal_sample (Identifier* mean, T stdev)
{ return normal_sample(mean, Identifier*(Constant::get<T>(stdev))); }

template <typename T>
Identifier* normal_sample (T mean, Identifier* stdev)
{ return normal_sample(Identifier*(Constant::get<T>(mean)), stdev); }

//! transpose, default perm is same as behavior n-1 ... 0
Identifier* transpose (Identifier* a);

Identifier* transpose (Identifier* a, Identifier* perm);

Identifier* transpose (Identifier* a, std::vector<uint64_t> perm);

//! flip a in specified dimensions
Identifier* flip (Identifier* a, Identifier* dims);

Identifier* flip (Identifier* a, std::vector<uint64_t> dims);

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

//! obtain the mean of a, lack of or invalid dimension look across all of a
Identifier* reduce_mean (Identifier* a);

Identifier* reduce_mean (Identifier* a, Identifier* dim);

Identifier* reduce_mean (Identifier* a, uint64_t dim);

//! obtain the l2norm of a, lack of or invalid dimension look across all of a
Identifier* reduce_l2norm (Identifier* a);

Identifier* reduce_l2norm (Identifier* a, Identifier* dim);

Identifier* reduce_l2norm (Identifier* a, uint64_t dim);

//! get the number of elements in a
Identifier* n_elems (Identifier* a);

//! get the number of elements in across a dimension in a
Identifier* n_dimension (Identifier* a, Identifier* dim);

Identifier* n_dimension (Identifier* a, uint64_t dim);

//! repeat a n times along inserted dimension dim
Identifier* expand (Identifier* a, Identifier* n, Identifier* dim);

Identifier* expand (Identifier* a, Identifier* n, uint64_t dim);

Identifier* expand (Identifier* a, uint64_t n, uint64_t dim);

//! clip values in range [min, max]
Identifier* clip (Identifier* a, Identifier* min, Identifier* max);

template <typename T>
Identifier* clip (Identifier* a, T min, T max)
{ return clip(a, Identifier*(Constant::get<T>(min)), Identifier*(Constant::get<T>(max))); }

//! normalize clip values with capacity cap
Identifier* clip_norm (Identifier* a, Identifier* cap);

template <typename T>
Identifier* clip_norm (Identifier* a, T cap)
{ return clip_norm(a, Identifier*(Constant::get<T>(cap))); }

//! matrix multiplication (todo: expand to include matmul along other dimensions, currently {0, 1} only)
Identifier* matmul (Identifier* a, Identifier* b);

// unimplemented

//! for example: window {0, 1} gives output f[i, j, :] = sum(a[i:i+filtshape[0], j:j+filtshape[1], :] * filter)
//! whereas window {0,2} gives output f[i, :, j] = sum(a[i:i+filtshape[0], :, j:j+filtshape[1]] * filter)
Identifier* cross_corr2d (Identifier* a, Identifier* filter, std::pair<uint64_t,uint64_t> dims = {0, 1});

//! convolve a with filter, conv(a, filter, dims) = cross_conv(a, flip(filter), dims)
Identifier* conv2d (Identifier* a, Identifier* filter, std::pair<uint64_t,uint64_t> dims = {0, 1});

}

#endif /* TENNCOR_OPERATIONS_HPP */
