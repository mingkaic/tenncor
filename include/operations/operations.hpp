/*!
 *
 *  operations.hpp
 *  cnnet
 *
 *  Purpose:
 *  elementary operators that wraps
 *  nodes in operation node
 *  using element wise transfer functions
 *
 *  transform operators that wraps
 *  nodes in operation node
 *  that reshapes arguments
 *
 *  Created by Mingkai Chen on 2016-10-24.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/leaf/constant.hpp"

#pragma once
#ifndef TENNCOR_OPERATIONS_HPP
#define TENNCOR_OPERATIONS_HPP

namespace nnet
{

// standard operations, using a single connector
#ifndef TENNCOR_OP_STD_HPP
#define TENNCOR_OP_STD_HPP

// >>>>>>>>>>>> UNARY STANDARD OPS <<<<<<<<<<<<

//! absolute value of a
varptr abs (const varptr a);

//! negative value of a
varptr operator - (const varptr a);

varptr operator ! (const varptr a);

//! sin of a
varptr sin (const varptr a);

//! cos of a
varptr cos (const varptr a);

//! tan of a
varptr tan (const varptr a);

//! cosecant of a
varptr csc (const varptr a);

//! secant of a
varptr sec (const varptr a);

//! cotangent of a
varptr cot (const varptr a);

//! e of power a
varptr exp (const varptr a);

//! natural log a
varptr log (const varptr a);

//! square root of a
varptr sqrt (const varptr a);

//! round a
varptr round (const varptr a);


// >>>>>>>>>>>> BINARY STANDARD OPS <<<<<<<<<<<<

//! b to the power of x
varptr pow (const varptr b, const varptr x);

//! add a and b
varptr operator + (const varptr a, const varptr b);

//! subtract a and b
varptr operator - (const varptr a, const varptr b);

//! multiply a and b
varptr operator * (const varptr a, const varptr b);

//! divide a and b
varptr operator / (const varptr a, const varptr b);

varptr operator == (const varptr a, const varptr b);

varptr operator != (const varptr a, const varptr b);

varptr operator < (const varptr a, const varptr b);

varptr operator > (const varptr a, const varptr b);

varptr binomial_sample (const varptr n, const varptr p);

varptr uniform_sample (const varptr min, const varptr max);

varptr normal_sample (const varptr mean, const varptr stdev);

template <typename T>
varptr pow (const varptr b, T x) { return pow(b, varptr(constant::get(x))); }

template <typename T>
varptr pow (T b, const varptr x) { return pow(varptr(constant::get(b)), x); }

template <typename T>
varptr operator + (const varptr a, T b) { return a + varptr(constant::get(b)); }

template <typename T>
varptr operator + (T a, const varptr b) { return varptr(constant::get(a)) + b; }

template <typename T>
varptr operator - (const varptr a, T b) { return a - varptr(constant::get(b)); }

template <typename T>
varptr operator - (T a, const varptr b) { return varptr(constant::get(a)) - b; }

template <typename T>
varptr operator * (const varptr a, T b) { return a * varptr(constant::get(b)); }

template <typename T>
varptr operator * (T a, const varptr b) { return varptr(constant::get(a)) * b; }

template <typename T>
varptr operator / (const varptr a, T b) { return a / varptr(constant::get(b)); }

template <typename T>
varptr operator / (T a, const varptr b) { return varptr(constant::get(a)) / b; }

template <typename T>
varptr operator == (const varptr a, T b) { return a == varptr(constant::get(b)); }

template <typename T>
varptr operator == (T a, const varptr b) { return varptr(constant::get(a)) == b; }

template <typename T>
varptr operator != (const varptr a, T b) { return a != varptr(constant::get(b)); }

template <typename T>
varptr operator != (T a, const varptr b) { return varptr(constant::get(a)) != b; }

template <typename T>
varptr binomial_sample (const varptr n, T p)
{ return binomial_sample(n, varptr(constant::get(p))); }

template <typename T>
varptr binomial_sample (T n, const varptr p)
{ return binomial_sample(varptr(constant::get(n)), p); }

template <typename T>
varptr uniform_sample (const varptr min, T max)
{ return uniform_sample(min, varptr(constant::get(max))); }

template <typename T>
varptr uniform_sample (T min, const varptr max)
{ return uniform_sample(varptr(constant::get(min)), max); }

template <typename T>
varptr normal_sample (const varptr mean, T stdev)
{ return normal_sample(mean, varptr(constant::get(stdev))); }

template <typename T>
varptr normal_sample (T mean, const varptr stdev)
{ return normal_sample(varptr(constant::get(mean)), stdev); }


// >>>>>>>>>>>> COORDINATE MAPPERS <<<<<<<<<<<<

//! transpose, default perm is same as behavior n-1 ... 0
varptr transpose (const varptr a, std::vector<size_t> perm = {});

//! flip a in specified dimensions
varptr flip (const varptr a, std::vector<size_t> dims);


// >>>>>>>>>>>> AGGREGATES <<<<<<<<<<<<

varptr arg_max (const varptr a);

varptr reduce_max (const varptr a);

varptr reduce_sum (const varptr a);


// >>>>>>>>>>>> SHAPE_DEP <<<<<<<<<<<<

varptr n_elems (const varptr a);

#endif /* TENNCOR_OP_STD_HPP */



// composite operations, using multiple connectors
#ifndef TENNCOR_OP_COM_HPP
#define TENNCOR_OP_COM_HPP

// >>>>>>>>>>>> CLIPPING <<<<<<<<<<<<

//! clip values in range [min, max]
varptr clip (const varptr a, const varptr min, const varptr max);

//! normalize clip values with capacity cap
varptr clip_norm (const varptr a, const varptr cap);

//! clip values in range [min, max]
template <typename T>
varptr clip (const varptr a, T min, T max)
{ return clip(a, varptr(constant::get(min)), varptr(constant::get(max))); }

//! normalize clip values with capacity cap
template <typename T>
varptr clip_norm (const varptr a, T cap)
{ return clip_norm(a, varptr(constant::get(cap))); }


// >>>>>>>>>>>> AGGREGATE <<<<<<<<<<<<

varptr reduce_mean (const varptr a);

varptr reduce_l2norm (const varptr a);


// >>>>>>>>>>>> MULTIPLEXED <<<<<<<<<<<<

// Dimensionality Reduction Functions (Wrappers for compress)
//! compress tensor by taking maximum value across specified dimension
//! unspecified dimension obtains maximum value in the entire tensor
varptr reduce_max (const varptr a, size_t dimension);

//! compress tensor by taking the sum of values across specified dimension(s)
//! unspecified dimension obtains the sum of all values in the entire tensor
varptr reduce_sum (const varptr a, size_t dimension);

//! compress tensor by taking the mean of values across specified dimension(s)
//! unspecified dimension obtains the mean of values in the entire tensor
varptr reduce_mean (const varptr a, size_t dimension);

varptr reduce_l2norm (const varptr a, size_t dimension);

//! obtains the indices of the maximum value across specified dimension
//! -1 index looks returns a vector coordinate specifying max value in tensor a
varptr arg_max (const varptr a, size_t dimension);

// unimplemented

//! for example: window {0, 1} gives output f[i, j, :] = sum(a[i:i+filtshape[0], j:j+filtshape[1], :] * filter)
//! whereas window {0,2} gives output f[i, :, j] = sum(a[i:i+filtshape[0], :, j:j+filtshape[1]] * filter)
//! if pad == true, then pad output with zero to fit a's shape, otherwise leave as is after cross_corr
varptr cross_corr2d (const varptr a, const varptr filter, std::pair<size_t,size_t> dims = {0, 1});

//! convolve a with filter, conv(a, filter, dims) = cross_conv(a, flip(filter), dims)
varptr conv2d (const varptr a, const varptr filter, std::pair<size_t,size_t> dims = {0, 1});

#endif /* TENNCOR_OP_COM_HPP */

#ifndef TENNCOR_OP_MATMUL_HPP
#define TENNCOR_OP_MATMUL_HPP

//! matrix multiplication (todo: expand to include matmul along other dimensions, currently {0, 1} only)
varptr matmul (const varptr a, const varptr b);

#endif /* TENNCOR_OP_MATMUL_HPP */

}

#endif /* TENNCOR_OPERATIONS_HPP */
