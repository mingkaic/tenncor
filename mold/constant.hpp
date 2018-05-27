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

	Constant (const Constant& other) : state_(other.state_)
	{
		size_t nbytes = state_.shape_.n_elems() * clay::type_size(state_.dtype_);
		data_ = clay::make_char(nbytes);
		std::memcpy(data_.get(), other.data_.get(), nbytes);
		state_.data_ = data_;
	}

	// >>>> CAN'T COPY OR MOVE (GOES AGAINST IMMUTABILITY) <<<<

	//! deleted move constructor
	Constant (Constant&&) = delete;

	//! copy assignment deleted
	Constant& operator = (const Constant&) = delete;

	//! move assignment deleted
	Constant& operator = (Constant&&) = delete;


	bool has_data (void) const override;

	clay::State get_state (void) const override;

protected:
	iNode* clone_impl (void) const override
	{
		return new Constant(*this);
	}

private:
	clay::State state_;

	std::shared_ptr<char> data_;
};

}

#endif /* MOLD_CONSTANT_HPP */
