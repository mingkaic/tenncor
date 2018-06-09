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

#include "mold/inode.hpp"

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

	Constant (const Constant& other);

	// >>>> CAN'T COPY OR MOVE (GOES AGAINST IMMUTABILITY) <<<<

	//! deleted move constructor
	Constant (Constant&&) = delete;

	//! copy assignment deleted
	Constant& operator = (const Constant&) = delete;

	//! move assignment deleted
	Constant& operator = (Constant&&) = delete;


	bool has_data (void) const override;

	clay::Shape get_shape (void) const override
	{
		return state_.shape_;
	}

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
