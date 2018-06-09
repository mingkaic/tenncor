/*!
 *
 *  unif_init.hpp
 *  kiln
 *
 *  Purpose:
 *  built a tensor with a uniform distributed values
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/tensor.hpp"
#include "clay/error.hpp"

#include "kiln/const_init.hpp"

#pragma once
#ifndef KILN_UNIF_INIT_HPP
#define KILN_UNIF_INIT_HPP

namespace kiln
{

clay::TensorPtrT unif_build (char* min, char* max,
	clay::Shape shape, clay::DTYPE dtype);

template <typename T>
clay::BuildTensorT unif_init (T min, T max,
	clay::Shape shape = clay::Shape())
{
	clay::DTYPE dtype = clay::get_type<T>();
	if (dtype == clay::DTYPE::BAD)
	{
		throw clay::UnsupportedTypeError(dtype);
	}
	correct_shape(shape, 1);
	return [min, max, shape, dtype]()
	{
		return unif_build((char*) &min, (char*) &max, shape, dtype);
	};
}

}

#endif /* KILN_UNIF_INIT_HPP */
