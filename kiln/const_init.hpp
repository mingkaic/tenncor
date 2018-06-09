/*!
 *
 *  const_init.hpp
 *  kiln
 *
 *  Purpose:
 *  built a tensor by a constant value or vector
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/tensor.hpp"
#include "clay/error.hpp"

#pragma once
#ifndef KILN_CONST_INIT_HPP
#define KILN_CONST_INIT_HPP

namespace kiln
{

void copy_over (char* dest, size_t ndest,
	const char* src, size_t nsrc);

void correct_shape (clay::Shape& shape, size_t n);

template <typename T>
clay::BuildTensorF const_init (T value,
	clay::Shape shape = clay::Shape())
{
	clay::DTYPE dtype = clay::get_type<T>();
	if (dtype == clay::DTYPE::BAD)
	{
		throw clay::UnsupportedTypeError(dtype);
	}
	correct_shape(shape, 1);
	return [value, shape, dtype]()
	{
		auto out = std::make_unique<clay::Tensor>(shape, dtype);
		char* dest = out->get_state().data_.lock().get();
		size_t nbytes = out->total_bytes();
		copy_over(dest, nbytes, (char*) &value, clay::type_size(dtype));
		return out;
	};
}

template <typename T>
clay::BuildTensorF const_init (std::vector<T> value,
	clay::Shape shape = clay::Shape())
{
	clay::DTYPE dtype = clay::get_type<T>();
	if (dtype == clay::DTYPE::BAD)
	{
		throw clay::UnsupportedTypeError(dtype);
	}
	correct_shape(shape, value.size());
	return [value, shape, dtype]()
	{
		auto out = std::make_unique<clay::Tensor>(shape, dtype);
		char* dest = out->get_state().data_.lock().get();
		size_t nbytes = out->total_bytes();
		copy_over(dest, nbytes, (char*) &value[0],
			value.size() * clay::type_size(dtype));
		return out;
	};
}

}

#endif /* KILN_CONST_INIT_HPP */
