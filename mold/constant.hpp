/*!
 *
 *  constant.hpp
 *  mold
 *
 *  Purpose:
 *  immutable implementation of inode
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <cassert>

#include "clay/memory.hpp"

#include "mold/functor.hpp"

#pragma once
#ifndef MOLD_CONSTANT_HPP
#define MOLD_CONSTANT_HPP

namespace mold
{

class Constant final : public iNode
{
public:
	Constant (std::shared_ptr<char> data,
		clay::Shape shape, clay::DTYPE type);

	// >>>> CAN'T COPY OR MOVE (GOES AGAINST SHARING) <<<<

	//! deleted copy constructor
	Constant (const Constant&) = delete;

	//! deleted move constructor
	Constant (Constant&&) = delete;

	//! copy assignment deleted
	Constant& operator = (const Constant&) = delete;

	//! move assignment deleted
	Constant& operator = (Constant&&) = delete;


	bool has_data (void) const override;

	clay::State get_state (void) const override;

	iNode* derive (iNode* wrt) override;

private:
	clay::State state_;

	std::shared_ptr<char> data_;
};

template <typename T>
iNode* make_constant (T scalar)
{
	std::shared_ptr<char> ptr = clay::make_char(sizeof(T));
	memcpy(ptr.get(), (char*) &scalar, sizeof(T));
	return new Constant(ptr, std::vector<size_t>{1}, clay::get_type<T>());
}

template <typename T>
iNode* make_constant (std::vector<T> vec, clay::Shape shape)
{
	size_t n = vec.size();
	assert(shape.n_elems() == n);
	size_t nbytes = sizeof(T) * n;
	std::shared_ptr<char> ptr = clay::make_char(nbytes);
	memcpy(ptr.get(), (char*) &vec[0], nbytes);
	return new Constant(ptr, shape, clay::get_type<T>());
}

}

#endif /* MOLD_CONSTANT_HPP */
