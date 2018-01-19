/*!
 *
 *  tens_elem_uni.hpp
 *  cnnet
 *
 *  Purpose:
 *  unary elementary actors
 *
 *  Created by Mingkai Chen on 2018-01-16.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/tensor/tensor_actor.hpp"

#pragma once
#ifndef TENNCOR_TENS_ELEM_UNI_HPP
#define TENNCOR_TENS_ELEM_UNI_HPP

namespace nnet
{

template <typename T>
struct tens_pipein : public tens_elem_uni<T>
{
	tens_pipein (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return data; }) {}
};

template <typename T>
struct tens_abs : public tens_elem_uni<T>
{
	tens_abs (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return std::abs(data); }) {}
};

template <typename T>
struct tens_neg : public tens_elem_uni<T>
{
	tens_neg (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return -data; }) {}
};

template <typename T>
struct tens_sin : public tens_elem_uni<T>
{
	tens_sin (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return std::sin(data); }) {}
};

template <typename T>
struct tens_cos : public tens_elem_uni<T>
{
	tens_cos (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return std::cos(data); }) {}
};

template <typename T>
struct tens_tan : public tens_elem_uni<T>
{
	tens_tan (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return std::tan(data); }) {}
};

template <typename T>
struct tens_csc : public tens_elem_uni<T>
{
	tens_csc (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return 1 / std::sin(data); }) {}
};

template <typename T>
struct tens_sec : public tens_elem_uni<T>
{
	tens_sec (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return 1 / std::cos(data); }) {}
};

template <typename T>
struct tens_cot : public tens_elem_uni<T>
{
	tens_cot (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return std::cos(data) / std::sin(data); }) {}
};

template <typename T>
struct tens_exp : public tens_elem_uni<T>
{
	tens_exp (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return std::exp(data); }) {}
};

template <typename T>
struct tens_sqrt : public tens_elem_uni<T>
{
	tens_sqrt (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return std::sqrt(data); }) {}
};

template <typename T>
struct tens_round : public tens_elem_uni<T>
{
	tens_round (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return std::round(data); }) {}
};

template <typename T>
struct tens_log : public tens_elem_uni<T>
{
	tens_log (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [](T data) { return std::log(data); }) {}
};

template <typename T>
struct tens_pow : public tens_elem_uni<T>
{
	tens_pow (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, double scalar) :
	tens_elem_uni<T>(dest, srcs, [scalar](T data) { return std::pow(data, scalar); }) {}
};

template <typename T>
struct tens_clip : public tens_elem_uni<T>
{
	tens_clip (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		double min, double max) :
	tens_elem_uni<T>(dest, srcs, [min, max](T data) -> T
	{
		if (min > data) return min;
		else if (max < data) return max;
		return data;
	}) {}
};

template <typename T>
struct tens_clip_norm : public tens_elem_uni<T>
{
	tens_clip_norm (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		const void* l2norm, double cap) :
	tens_elem_uni<T>(dest, srcs, [l2norm, cap](T data) -> T
	{
		T l2 = *((const T*) l2norm);
		if (l2 > cap)
		{
			return data * cap / l2;
		}
		return data;
	}) {}
};

template <typename T>
struct tens_bin_sample_uni : public tens_elem_uni<T>
{
	tens_bin_sample_uni (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		signed n) :
	tens_elem_uni<T>(dest, srcs, [n](T data)
	{
		assert(data>= 0 && data <= 1);
		std::binomial_distribution<int> dist(n, data);
		return dist(nnutils::get_generator());
	}) {}

	tens_bin_sample_uni (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		double p) :
	tens_elem_uni<T>(dest, srcs, [p](T data)
	{
		assert(p>= 0 && p <= 1);
		std::binomial_distribution<int> dist(data, p);
		return dist(nnutils::get_generator());
	}) {}
};

template <typename T>
struct tens_uni_add : public tens_elem_uni<T>
{
	tens_uni_add (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		double b) :
	tens_elem_uni<T>(dest, srcs, [b](T data)
	{
		return data + b;
	}) {}
};

template <typename T>
struct tens_uni_sub : public tens_elem_uni<T>
{
	tens_uni_sub (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		double b) :
	tens_elem_uni<T>(dest, srcs, [b](T data)
	{
		return data - b;
	}) {}

	tens_uni_sub (double a, out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [a](T data)
	{
		return a - data;
	}) {}
};

template <typename T>
struct tens_uni_mul : public tens_elem_uni<T>
{
	tens_uni_mul (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		double b) :
	tens_elem_uni<T>(dest, srcs, [b](T data)
	{
		return data * b;
	}) {}
};

template <typename T>
struct tens_uni_div : public tens_elem_uni<T>
{
	tens_uni_div (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		double b) :
	tens_elem_uni<T>(dest, srcs, [b](T data)
	{
		return data / b;
	}) {}

	tens_uni_div (double a, out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_elem_uni<T>(dest, srcs, [a](T data)
	{
		return a / data;
	}) {}
};

}

#endif /* TENNCOR_TENS_ELEM_UNI_HPP */
