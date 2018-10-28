///
///	grader.hpp
///	ade
///
///	Purpose:
///	Define derivative chain rules and map to OPCODEs
///

#include "ade/opcode.hpp"
#include "ade/tensor.hpp"
#include "ade/coord.hpp"

#ifndef ADE_GRADER_HPP
#define ADE_GRADER_HPP

namespace ade
{

// TODO: CONVERT TO GENERATED CONFIG

using ArgsT = std::vector<std::pair<CoordPtrT,Tensorptr>>;

template <OPCODE OP>
Tensorptr grader (ArgsT& args, ArgsT& grads)
{
	throw std::bad_function_call();
}

#define GRAD_DECLARE(CODE)template <>\
Tensorptr grader<CODE> (ArgsT&,ArgsT&);

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

Tensorptr gradmap (OPCODE op, ArgsT args, ArgsT& grads);

}

#endif // ADE_GRADER_HPP
