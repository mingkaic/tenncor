/*!
 *
 *  tens_transform.hpp
 *  cnnet
 *
 *  Purpose:
 *  transformation actors
 *
 *  Created by Mingkai Chen on 2018-01-16.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/tensor/tensor_actor.hpp"

#pragma once
#ifndef TENNCOR_TENS_TRANSFORM_HPP
#define TENNCOR_TENS_TRANSFORM_HPP

namespace nnet
{

template <typename T>
struct tens_l2norm : public tens_general<T>
{
	tens_l2norm (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_transpose : public tens_general<T>
{
	tens_transpose (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs, 
		std::pair<size_t, size_t> axis_swap);
};

template <typename T>
struct tens_fit : public tens_general<T>
{
	tens_fit (out_wrapper<void> dest,
		std::vector<in_wrapper<void> > srcs);
};

template <typename T>
struct tens_extend : public tens_general<T>
{
	tens_extend (out_wrapper<void> dest,
		std::vector<in_wrapper<void> > srcs, 
		size_t index, size_t multiplier);
};

template <typename T>
struct tens_compress : public tens_general<T>
{
	tens_compress (out_wrapper<void> dest,
		std::vector<in_wrapper<void> > srcs, 
		size_t index, BI_TRANS<T> collector);

	tens_compress (out_wrapper<void> dest,
		std::vector<in_wrapper<void> > srcs, 
		BI_TRANS<T> collector);
};

template <typename T>
struct tens_argcompress : public tens_general<T>
{
	tens_argcompress (out_wrapper<void> dest,
		std::vector<in_wrapper<void> > srcs, 
		size_t dimension, REDUCE<T> search);

	tens_argcompress (out_wrapper<void> dest,
		std::vector<in_wrapper<void> > srcs, 
		REDUCE<T> search);
};

template <typename T>
struct tens_flip : public tens_general<T>
{
	tens_flip (out_wrapper<void> dest,
		std::vector<in_wrapper<void> > srcs, 
		std::vector<size_t> dims);
};

template <typename T>
struct tens_cross_corr2d : public tens_general<T>
{
	tens_cross_corr2d (out_wrapper<void> dest,
		std::vector<in_wrapper<void> > srcs,
		std::pair<size_t, size_t> dims);
};

}

#include "src/tensor/actors/tens_transform.ipp"

#endif /* TENNCOR_TENS_TRANSFORM_HPP */
