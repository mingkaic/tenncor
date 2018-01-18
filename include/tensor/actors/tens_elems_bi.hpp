/*!
 *
 *  tens_elems_bi.hpp
 *  cnnet
 *
 *  Purpose:
 *  binary elementary actors
 *
 *  Created by Mingkai Chen on 2018-01-16.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/tensor/tensor_actor.hpp"

#pragma once
#ifndef TENNCOR_TENS_ELEMS_BI_HPP
#define TENNCOR_TENS_ELEMS_BI_HPP

namespace nnet
{

template <typename T>
struct tens_bin_sample : public tens_elems_bi<T>
{
	tens_bin_sample (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elems_bi<T>(dest, srcs, [](T n, T p)
	{
		assert(p>= 0 && p <= 1);
		std::binomial_distribution<T> dist(n, p);
		return dist(nnutils::get_generator());
	}) {}
};

template <typename T>
struct tens_add : public tens_elems_bi<T>
{
	tens_add (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elems_bi<T>(dest, srcs, [](T a, T b)
	{ return a + b; }) {}
};

template <typename T>
struct tens_sub : public tens_elems_bi<T>
{
	tens_sub (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elems_bi<T>(dest, srcs, [](T a, T b)
	{ return a - b; }) {}
};

template <typename T>
struct tens_mul : public tens_elems_bi<T>
{
	tens_mul (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elems_bi<T>(dest, srcs, 
	[](T a, T b) { return a * b; }) {}
};

template <typename T>
struct tens_div : public tens_elems_bi<T>
{
	tens_div (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elems_bi<T>(dest, srcs, 
	[](T a, T b) { return a / b; }) {}
};

template <typename T>
struct tens_axial_add : public tens_axial_elems<T>
{
	tens_axial_add (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		size_t axis) :
	tens_axial_elems<T>(dest, srcs, axis, true, 
	[](T a, T b) { return a + b; }) {}
};

template <typename T>
struct tens_axial_sub : public tens_axial_elems<T>
{
	tens_axial_sub (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		size_t axis, bool left) :
	tens_axial_elems<T>(dest, srcs, axis, left, 
	[](T a, T b) { return a - b; }) {}
};

template <typename T>
struct tens_axial_mul : public tens_axial_elems<T>
{
	tens_axial_mul (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		size_t axis) :
	tens_axial_elems<T>(dest, srcs, axis, true, 
	[](T a, T b) { return a * b; }) {}
};

template <typename T>
struct tens_axial_div : public tens_axial_elems<T>
{
	tens_axial_div (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		size_t axis, bool left) :
	tens_axial_elems<T>(dest, srcs, axis, left, 
	[](T a, T b) { return a / b; }) {}
};

}

#endif /* TENNCOR_TENS_ELEMS_BI_HPP */
