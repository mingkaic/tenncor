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
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/immutable/immutable.hpp"
#include "include/graph/leaf/constant.hpp"
#include "include/graph/operations/operation_utils.hpp"
#include "include/tensor/actors/tens_matmul.hpp"

#pragma once

namespace nnet
{

#ifndef TENNCOR_ELEM_UNI_HPP
#define TENNCOR_ELEM_UNI_HPP

//! wraps an empty node, usually to avoid overlapping references
varptr identity (varptr x);

varptr as_constant (varptr x);

//! absolute value of a
varptr operator + (const varptr a);

//! negative value of a
varptr operator - (const varptr a);

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

//! square root of a
varptr sqrt (const varptr a);

//! round a
varptr round (const varptr a);

//! natural log a
varptr log (const varptr a);

//! a to the power of scalar
varptr pow (const varptr a, double scalar);

//! clip values in range [min, max]
varptr clip (const varptr a, double min, double max);

//! normalize clip values with capacity cap
varptr clip_norm (const varptr a, double cap);

//! output value 0 if false == compare(a, b) else 1 for each element
varptr conditional (double a, const varptr b, std::function<bool(double,double)> compare, std::string name);

varptr conditional (const varptr a, double b, std::function<bool(double,double)> compare, std::string name);

varptr eq (double a, const varptr b);

varptr eq (const varptr a, double b);

varptr neq (double a, const varptr b);

varptr neq (const varptr a, double b);

//! sample using binominal distribution given tensors (or scalars) n and p
// todo: after implementing type, restrict n to integers
varptr binomial_sample (signed n, const varptr p);

varptr binomial_sample (const varptr n, double p);

//! add a and scalar b
varptr operator + (const varptr a, double b);

//! add scalar a and b
varptr operator + (double a, const varptr b);

//! subtract a and scalar b
varptr operator - (const varptr a, double b);

//! subtract scalar a and b
varptr operator - (double a, const varptr b);

//! multiply a and scalar b
varptr operator * (const varptr a, double b);

//! multiply scalar a and b
varptr operator * (double a, const varptr b);

//! divide a and scalar b
varptr operator / (const varptr a, double b);

//! divide scalar a and b
varptr operator / (double a, const varptr b);

#endif /* TENNCOR_ELEM_UNI_HPP */

#ifndef TENNCOR_ELEM_BI_HPP
#define TENNCOR_ELEM_BI_HPP

varptr binomial_sample (const varptr n, const varptr p);

varptr conditional (const varptr a, const varptr b, std::function<bool(double,double)> compare, std::string name);

varptr eq (const varptr a, const varptr b);

varptr neq (const varptr a, const varptr b);

//! add a and b
varptr operator + (const varptr a, const varptr b);

//! subtract a and b
varptr operator - (const varptr a, const varptr b);

//! multiply a and b
varptr operator * (const varptr a, const varptr b);

//! divide a and b
varptr operator / (const varptr a, const varptr b);

// START DEPRECATE
//! add a and b along a specific axis, dimension values outside of axis must match
varptr add_axial_a (const varptr a, const varptr b, size_t axis_a);

varptr add_axial_b (const varptr a, const varptr b, size_t axis_b);

//! subtract a and b along a specific axis, dimension values outside of axis must match
varptr sub_axial_a (const varptr a, const varptr b, size_t axis_a);

varptr sub_axial_b (const varptr a, const varptr b, size_t axis_b);

//! multiply a and b along a specific axis, dimension values outside of axis must match
varptr mul_axial_a (const varptr a, const varptr b, size_t axis_a);

varptr mul_axial_b (const varptr a, const varptr b, size_t axis_b);

//! divide a and b along a specific axis, dimension values outside of axis must match
varptr div_axial_a (const varptr a, const varptr b, size_t axis_a);

varptr div_axial_b (const varptr a, const varptr b, size_t axis_b);
// END DEPRECATE

#endif /* TENNCOR_ELEM_BI_HPP */

#ifndef TENNCOR_TRANSFORM_HPP
#define TENNCOR_TRANSFORM_HPP

varptr l2norm (const varptr a);

//! transpose a along first 2 dimension
// todo: check if axis_swap are the same dimensions, if so, return a as is (invalid transpose) + leave warning
varptr transpose (const varptr a, std::pair<size_t,size_t> axis_swap = {0, 1});

//! fit data in a to watch's shape, ignores all jacobian (todo: change to selectively ignore watch's jacobian)
//! watch needs to be a dependency of the resulting node,
//! because shape changes to watch should trigger shape update for output node
varptr fit (const varptr a, const varptr watch);

//! extend data in a to along index dimension multiplier times
varptr extend (const varptr a, size_t index, size_t multiplier);

//! compresses data along dimensions specified by index
//! unspecified index compresses all elements in the tensor (output is a scalar)
varptr compress (const varptr a, BI_TRANS<double> collector,
	optional<size_t> index, std::string name = "compress");

// Dimensionality Reduction Functions (Wrappers for compress)
//! compress tensor by taking maximum value across specified dimension
//! unspecified dimension obtains maximum value in the entire tensor
varptr reduce_max (const varptr a, optional<size_t> dimension = optional<size_t>());

//! compress tensor by taking the sum of values across specified dimension(s)
//! unspecified dimension obtains the sum of all values in the entire tensor
varptr reduce_sum (const varptr a, optional<size_t> dimension = optional<size_t>());

//! compress tensor by taking the mean of values across specified dimension(s)
//! unspecified dimension obtains the mean of values in the entire tensor
varptr reduce_mean (const varptr a, optional<size_t> dimension = optional<size_t>());

//! compresses data along dimensions specified by dimension
//! by taking the index using the compare function
//! unspecified dimension compresses all elements in the tensor (output is a scalar)
//! takes left argument of compare if compare evaluates to true
varptr arg_compress (const varptr a, REDUCE<double> search,
	optional<size_t> dimension, std::string name = "argcompress");

//! obtains the indices of the maximum value across specified dimension
//! -1 index looks returns a vector coordinate specifying max value in tensor a
varptr arg_max (const varptr a, optional<size_t> dimension = optional<size_t>());

//! flip a in specified dimensions
varptr flip (const varptr a, std::vector<size_t> dims);

//! for example: window {0, 1} gives output f[i, j, :] = sum(a[i:i+filtshape[0], j:j+filtshape[1], :] * filter)
//! whereas window {0,2} gives output f[i, :, j] = sum(a[i:i+filtshape[0], :, j:j+filtshape[1]] * filter)
//! if pad == true, then pad output with zero to fit a's shape, otherwise leave as is after cross_corr
varptr cross_corr2d (const varptr a, const varptr filter, std::pair<size_t,size_t> dims = {0, 1});
	
//! convolve a with filter, conv(a, filter, dims) = cross_conv(a, flip(filter), dims)
varptr conv2d (const varptr a, const varptr filter, std::pair<size_t,size_t> dims = {0, 1});

// todo: implement
// [grad(trace(f(x)), x) = transpose(scalar_grad(f(x), x))]
//! trace of a
varptr trace (const varptr a);

//! inverse of matrix a
varptr inverse (const varptr a);

#endif /* TENNCOR_TRANSFORM_HPP */

#ifndef TENNCOR_MATMUL_HPP
#define TENNCOR_MATMUL_HPP

//! matrix multiplication (todo: expand to include matmul along other dimensions, currently {0, 1} only)
varptr matmul (const varptr a, const varptr b,
	bool transposeA = false, bool transposeB = false);

#endif /* TENNCOR_MATMUL_HPP */

}
