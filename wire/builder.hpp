/*!
 *
 *  builder.hpp
 *  wire
 *
 *  Purpose:
 *  help build inode instances
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <cassert>

#include "mold/constant.hpp"

#pragma once
#ifndef WIRE_CONSTANT_HPP
#define WIRE_CONSTANT_HPP

namespace wire
{

template <typename T>
mold::iNode* get_constant (T scalar)
{
	std::shared_ptr<char> ptr = clay::make_char(sizeof(T));
	memcpy(ptr.get(), (char*) &scalar, sizeof(T));
	return new mold::Constant(ptr, std::vector<size_t>{1}, clay::get_type<T>());
}

template <typename T>
mold::iNode* get_constant (std::vector<T> vec, clay::Shape shape)
{
	size_t n = vec.size();
	assert(shape.n_elems() == n);
	size_t nbytes = sizeof(T) * n;
	std::shared_ptr<char> ptr = clay::make_char(nbytes);
	memcpy(ptr.get(), (char*) &vec[0], nbytes);
	return new mold::Constant(ptr, shape, clay::get_type<T>());
}

}

#endif /* WIRE_CONSTANT_HPP */
