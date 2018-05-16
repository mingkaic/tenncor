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

Constant* make_one (clay::DTYPE dtype);

}

#endif /* MOLD_CONSTANT_HPP */
