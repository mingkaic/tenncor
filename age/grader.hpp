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

namespace ade
{

/// Return a Tensor::SYMBOLIC_ONE extended to input shape
Tensorptr shaped_one (Shape shape);

/// Return a Tensor::SYMBOLIC_ZERO extended to input shape
Tensorptr shaped_zero (Shape shape);

// TODO: CONVERT TO GENERATED CONFIG

// todo: remove and make better
#define MAKE_CODE(CODE)ade::CodePtrT(new ade::Opcode<CODE>)

template <OPCODE OP>
struct Opcode final : public iOpcode
{
	std::string to_string (void) const override
	{
		return ade::opname(OP);
	}

	size_t opnum (void) const override
	{
		return OP;
	}

	Tensorptr gradient (ArgsT args, size_t gradidx) const override
	{
		throw std::bad_function_call();
	}

	Tensorptr grad_vertical_merge (MappedTensor bot, MappedTensor top) const override
	{
		return Functor::get(MAKE_CODE(MUL), {
			{identity, Functor::get(MAKE_CODE(ADD), {bot})}, top,
		});
	}

	Tensorptr grad_horizontal_merge (ArgsT& grads) const override
	{
		return Functor::get(MAKE_CODE(ADD), grads);
	}
};

ade::CodePtrT make_code (OPCODE opcode);

#define GRAD_DECLARE(CODE)template <>\
Tensorptr Opcode<CODE>::gradient (ade::ArgsT args, size_t gradidx) const;

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
