/*!
 *
 *  functor.hpp
 *  mold
 *
 *  Purpose:
 *  functor implementation of inode
 *  performs forward operations when necessary
 *  create gradient nodes when called
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "mold/inode.hpp"

#pragma once
#ifndef MOLD_OPERATE_IO_HPP
#define MOLD_OPERATE_IO_HPP

namespace mold
{

using ImmPair = std::pair<clay::Shape,clay::DTYPE>;

class iOperateIO
{
public:
	virtual ~iOperateIO (void) = default;

	iOperateIO* clone (void) const
	{
		return clone_impl();
	}

	virtual ImmPair get_imms (
		std::vector<clay::State> args) const = 0;

	virtual bool write_data (clay::State& dest,
		std::vector<clay::State> args) const = 0;

private:
	virtual iOperateIO* clone_impl (void) const = 0;
};

using iOperatePtrT = std::shared_ptr<iOperateIO>;

}

#endif /* MOLD_OPERATE_IO_HPP */