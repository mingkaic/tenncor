/*!
 *
 *  tens_matmul.hpp
 *  cnnet
 *
 *  Purpose:
 *  matrix multiplication actor
 *
 *  Created by Mingkai Chen on 2018-01-16.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/tensor/tensor_actor.hpp"

#pragma once
#ifndef TENNCOR_TENS_MATMUL_HPP
#define TENNCOR_TENS_MATMUL_HPP

#define STRASSEN_THRESHOLD 256

namespace nnet
{

template <typename T>
struct tens_matmul : public tens_general<T>
{
	tens_matmul (out_wrapper<void> dest,
		std::vector<in_wrapper<void> > srcs);
};

}

#include "src/tensor/actors/tens_matmul.ipp"

#endif /* TENNCOR_TENS_MATMUL_HPP */
