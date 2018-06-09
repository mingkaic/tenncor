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

using ShaperF = std::function<clay::Shape(std::vector<clay::State>)>;

using TyperF = std::function<clay::DTYPE(std::vector<clay::DTYPE>)>;

using ImmPair = std::pair<clay::Shape,clay::DTYPE>;

using TypeRegT = EnumMap<clay::DTYPE,ArgsF>;

class OperateIO final : public mold::iOperateIO
{
public:
	OperateIO (TypeRegT ops, ShaperF shaper, TyperF typer) :
		ops_(ops), shaper_(shaper), typer_(typer) {}

	bool validate_data (clay::State state,
		std::vector<clay::State> args) const override
	{
		auto imms = get_imms(args);
		return state.shape_.is_compatible_with(imms.first) &&
			state.dtype_ == imms.second;
	}

	bool write_data (clay::State& dest,
		std::vector<clay::State> args) const override
	{
		auto imms = get_imms(args);
		bool success = dest.shape_.
			is_compatible_with(imms.first) &&
			dest.dtype_ == imms.second;
		if (success)
		{
			unsafe_write(dest, args, imms.second);
		}
		return success;
	}

	clay::TensorPtrT make_data (
		std::vector<clay::State> args) const override
	{
		auto imms = get_imms(args);
		clay::Shape& shape = imms.first;
		clay::DTYPE& dtype = imms.second;
		clay::Tensor* out = new clay::Tensor(shape, dtype);
		clay::State dest = out->get_state();
		unsafe_write(dest, args, dtype);
		return clay::TensorPtrT(out);
	}

private:
	ImmPair get_imms (std::vector<clay::State>& args) const
	{
		if (args.empty())
		{
			throw NoArgumentsError();
		}
		std::vector<clay::DTYPE> types(args.size());
		std::transform(args.begin(), args.end(), types.begin(),
		[](clay::State& state) -> clay::DTYPE
		{
			return state.dtype_;
		});
		clay::DTYPE otype = typer_(types);
		return {shaper_(args), otype};
	}

	void unsafe_write (clay::State& dest,
		std::vector<clay::State>& args, clay::DTYPE dtype) const
	{
		auto op = ops_.find(dtype);
		if (ops_.end() == op)
		{
			throw clay::UnsupportedTypeError(dtype);
		}
		op->second(dest, args);
	}

	iOperateIO* clone_impl (void) const override
	{
		return new OperateIO(*this);
	}

	TypeRegT ops_;

	ShaperF shaper_;

	TyperF typer_;
};

}

#endif /* SLIP_OPERATE_IO_HPP */
