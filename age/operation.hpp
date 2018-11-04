///
///	operation.hpp
///	age
///
///	Purpose:
///	Define operation implementation for ade functors
///

#include "ade/functor.hpp"

#include "age/opcode.hpp"

#ifndef AGE_OPERATION_HPP
#define AGE_OPERATION_HPP

namespace age
{

#define MAKE_CODE(CODE)ade::OpPtrT(new age::Operation<CODE>)

template <OPCODE OP>
struct Operation final : public ade::iOperation
{
	std::string to_string (void) const override
	{
		return age::opname(OP);
	}

	size_t opnum (void) const override
	{
		return OP;
	}

	ade::Tensorptr gradient (ade::ArgsT args, size_t gradidx) const override
	{
		throw std::bad_function_call();
	}

	ade::Tensorptr grad_vertical_merge (ade::MappedTensor bot, ade::MappedTensor top) const override
	{
		return ade::Functor::get(MAKE_CODE(MUL), {
			{ade::identity, ade::Functor::get(MAKE_CODE(ADD), {bot})}, top,
		});
	}

	ade::Tensorptr grad_horizontal_merge (ade::ArgsT& grads) const override
	{
		return ade::Functor::get(MAKE_CODE(ADD), grads);
	}
};

ade::OpPtrT make_code (OPCODE opcode);

/// Return a Tensor::SYMBOLIC_ONE extended to input shape
ade::Tensorptr shaped_one (ade::Shape shape);

/// Return a Tensor::SYMBOLIC_ZERO extended to input shape
ade::Tensorptr shaped_zero (ade::Shape shape);

}

#endif // AGE_OPERATION_HPP
