#include "age/runtime/grader.hpp"
#include "llo/data.hpp"
#include "llo/helper.hpp"

#ifndef _GENERATED_RT_HPP
#define _GENERATED_RT_HPP

namespace age
{

enum _GENERATED_OPCODES
{
	ABS = 0,
	COS,
	DIV,
	EQ,
	EXP,
	GT,
	LOG,
	LT,
	MAX,
	MIN,
	NEG,
	NEQ,
	POW,
	PROD,
	RAND_BINO,
	RAND_NORM,
	RAND_UNIF,
	ROUND,
	SIN,
	SQRT,
	SUB,
	SUM,
	TAN,
};

template <typename T>ade::Tensor* data (T scalar, ade::Shape shape)
{
	return llo::get_variable(std::vector<T>(shape.n_elems(),scalar),shape,err::sprintf("%d",scalar));
}

ade::Opcode sum_opcode (void);

ade::Opcode prod_opcode (void);

_GENERATED_OPCODES nameop (std::string name);

std::string opname (_GENERATED_OPCODES code);

ade::Tensorptr grad_rule (size_t code,TensT args,size_t idx);

}

#endif
