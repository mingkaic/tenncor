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
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_abs : public tens_elem_uni<T>
{
	tens_abs (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_neg : public tens_elem_uni<T>
{
	tens_neg (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_sin : public tens_elem_uni<T>
{
	tens_sin (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_cos : public tens_elem_uni<T>
{
	tens_cos (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_tan : public tens_elem_uni<T>
{
	tens_tan (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_csc : public tens_elem_uni<T>
{
	tens_csc (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_sec : public tens_elem_uni<T>
{
	tens_sec (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_cot : public tens_elem_uni<T>
{
	tens_cot (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_exp : public tens_elem_uni<T>
{
	tens_exp (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_sqrt : public tens_elem_uni<T>
{
	tens_sqrt (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_round : public tens_elem_uni<T>
{
	tens_round (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_log : public tens_elem_uni<T>
{
	tens_log (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_pow : public tens_elem_uni<T>
{
	tens_pow (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, double scalar);
};

template <typename T>
struct tens_clip : public tens_elem_uni<T>
{
	tens_clip (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		double min, double max);
};

template <typename T>
struct tens_clip_norm : public tens_elem_uni<T>
{
	tens_clip_norm (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		const void* l2norm, double cap);
};

template <typename T>
struct tens_bin_sample_uni : public tens_elem_uni<T>
{
	tens_bin_sample_uni (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		signed n);

	tens_bin_sample_uni (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		double p);
};

template <typename T>
struct tens_uni_add : public tens_elem_uni<T>
{
	tens_uni_add (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		double b);
};

template <typename T>
struct tens_uni_sub : public tens_elem_uni<T>
{
	tens_uni_sub (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		double b);

	tens_uni_sub (double a, out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_uni_mul : public tens_elem_uni<T>
{
	tens_uni_mul (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		double b);
};

template <typename T>
struct tens_uni_div : public tens_elem_uni<T>
{
	tens_uni_div (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		double b);

	tens_uni_div (double a, out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

}

#include "src/tensor/actors/tens_elem_uni.ipp"

#endif /* TENNCOR_TENS_ELEM_UNI_HPP */
