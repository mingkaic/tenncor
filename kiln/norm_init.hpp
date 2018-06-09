/*!
 *
 *  norm_init.hpp
 *  kiln
 *
 *  Purpose:
 *  built a tensor with a normal distributed values
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/tensor.hpp"
#include "clay/error.hpp"

#include "kiln/const_init.hpp"

#pragma once
#ifndef KILN_NORM_INIT_HPP
#define KILN_NORM_INIT_HPP

namespace kiln
{

clay::TensorPtrT norm_build (char* mean, char* stdev,
	clay::Shape shape, clay::DTYPE dtype);

template <typename T>
clay::BuildTensorF norm_init (T mean, T stdev,
	clay::Shape shape = clay::Shape())
{
	clay::DTYPE dtype = clay::get_type<T>();
	if (dtype == clay::DTYPE::BAD)
	{
		throw clay::UnsupportedTypeError(dtype);
	}
	correct_shape(shape, 1);
	return [mean, stdev, shape, dtype]()
	{
		return norm_build((char*) &mean, (char*) &stdev, shape, dtype);
	};
}

}

#endif /* KILN_NORM_INIT_HPP */
