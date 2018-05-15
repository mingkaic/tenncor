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

using ArgsF = std::function<void(clay::State&,std::vector<clay::State>)>;

using ShaperF = std::function<clay::Shape(std::vector<clay::Shape>)>;

using TyperF = std::function<clay::DTYPE(std::vector<clay::DTYPE>)>;

struct OperateIO final : public clay::iSource
{
	OperateIO (ArgsF op, ShaperF shaper, TyperF typer);

	bool read_data (clay::State& dest) const override;

	clay::TensorPtrT get (void) const;

	std::vector<clay::State> args_;

private:
	std::pair<clay::Shape, clay::DTYPE> expect_out (void) const;

	ArgsF op_;
	
	ShaperF shaper_;
	
	TyperF typer_;
};

}

#endif /* MOLD_OPERATE_IO_HPP */
