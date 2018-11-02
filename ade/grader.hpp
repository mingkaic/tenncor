///
///	grader.hpp
///	ade
///
///	Purpose:
///	Define derivative chain rules and map to OPCODEs
///

#include "ade/ifunctor.hpp"

#ifndef ADE_GRADER_HPP
#define ADE_GRADER_HPP

namespace ade
{

// TODO: CONVERT TO GENERATED CONFIG

template <OPCODE OP>
Tensorptr grader (ArgsT args, size_t gradidx)
{
	throw std::bad_function_call();
}

#define GRAD_DECLARE(CODE)template <>\
Tensorptr grader<CODE> (ArgsT args, size_t gradidx);

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

Tensorptr gradmap (OPCODE op, ArgsT args, size_t gradidx);

}

#endif // ADE_GRADER_HPP
