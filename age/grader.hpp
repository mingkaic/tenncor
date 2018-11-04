///
///	grader.hpp
///	ade
///
///	Purpose:
///	Define derivative chain rules and map to OPCODEs
///

#include "ade/functor.hpp"

#include "age/opcode.hpp"

#ifndef ADE_GRADER_HPP
#define ADE_GRADER_HPP

namespace age
{

/// Return a Tensor::SYMBOLIC_ONE extended to input shape
ade::Tensorptr shaped_one (ade::Shape shape);

/// Return a Tensor::SYMBOLIC_ZERO extended to input shape
ade::Tensorptr shaped_zero (ade::Shape shape);

// TODO: CONVERT TO GENERATED CONFIG

// todo: remove and make better
#define MAKE_CODE(CODE)ade::CodePtrT(new age::Opcode<CODE>)

template <OPCODE OP>
struct Opcode final : public ade::iOpcode
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

ade::CodePtrT make_code (OPCODE opcode);

#define GRAD_DECLARE(CODE)template <>\
ade::Tensorptr Opcode<CODE>::gradient (ade::ArgsT args, size_t gradidx) const;

GRAD_DECLARE(COPY)

GRAD_DECLARE(ABS)

GRAD_DECLARE(NEG)

GRAD_DECLARE(SIN)

GRAD_DECLARE(COS)

GRAD_DECLARE(TAN)

GRAD_DECLARE(EXP)

GRAD_DECLARE(LOG)

GRAD_DECLARE(SQRT)

GRAD_DECLARE(ROUND)

GRAD_DECLARE(POW)

GRAD_DECLARE(ADD)

GRAD_DECLARE(SUB)

GRAD_DECLARE(MUL)

GRAD_DECLARE(DIV)

GRAD_DECLARE(EQ)

GRAD_DECLARE(NE)

GRAD_DECLARE(LT)

GRAD_DECLARE(GT)

GRAD_DECLARE(MIN)

GRAD_DECLARE(MAX)

GRAD_DECLARE(RAND_BINO)

GRAD_DECLARE(RAND_UNIF)

GRAD_DECLARE(RAND_NORM)

#undef GRAD_DECLARE

}

#endif // ADE_GRADER_HPP
