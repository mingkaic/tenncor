/*!
 *
 *  operate_io.hpp
 *  slip
 *
 *  Purpose:
 *  ioperate_io implementation generating tensor
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/state.hpp"
#include "clay/error.hpp"

#include "mold/ioperate_io.hpp"

#include "slip/opcode.hpp"
#include "slip/error.hpp"

#pragma once
#ifndef SLIP_OPERATE_IO_HPP
#define SLIP_OPERATE_IO_HPP

namespace slip
{

using ArgsF = std::function<void(clay::State&,std::vector<clay::State>)>;

using ShaperF = std::function<clay::Shape(std::vector<mold::StateRange>)>;

using TyperF = std::function<clay::DTYPE(std::vector<clay::DTYPE>)>;

using ImmPair = std::pair<clay::Shape,clay::DTYPE>;

using TypeRegT = EnumMap<clay::DTYPE,ArgsF>;

class OperateIO final : public mold::iOperateIO
{
public:
	OperateIO (TypeRegT ops, ShaperF shaper, TyperF typer);

	bool validate_data (clay::State state,
		std::vector<mold::StateRange> args) const override;

	bool write_data (clay::State& dest,
		std::vector<mold::StateRange> args) const override;

	clay::TensorPtrT make_data (
		std::vector<mold::StateRange> args) const override;

private:
	ImmPair get_imms (std::vector<mold::StateRange>& args) const;

	void unsafe_write (clay::State& dest,
		std::vector<mold::StateRange>& args, clay::DTYPE dtype) const;

	mold::iOperateIO* clone_impl (void) const override;

	TypeRegT ops_;

	ShaperF shaper_;

	TyperF typer_;
};

}

#endif /* SLIP_OPERATE_IO_HPP */
