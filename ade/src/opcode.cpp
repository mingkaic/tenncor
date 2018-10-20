#include <unordered_map>

#include "ade/opcode.hpp"

#ifdef ADE_OPCODE_HPP

namespace ade
{

struct EnumHash
{
	template <typename T>
	size_t operator() (T e) const
	{
		return static_cast<size_t>(e);
	}
};

#define OP_ASSOC(CODE) std::pair<OPCODE,std::string>{CODE, #CODE}

const std::unordered_map<OPCODE,std::string,EnumHash> opnames =
{
	OP_ASSOC(ABS),
	OP_ASSOC(NEG),
	OP_ASSOC(NOT),
	OP_ASSOC(SIN),
	OP_ASSOC(COS),
	OP_ASSOC(TAN),
	OP_ASSOC(EXP),
	OP_ASSOC(LOG),
	OP_ASSOC(SQRT),
	OP_ASSOC(ROUND),
	OP_ASSOC(FLIP),

	OP_ASSOC(POW),
	OP_ASSOC(ADD),
	OP_ASSOC(SUB),
	OP_ASSOC(MUL),
	OP_ASSOC(DIV),
	OP_ASSOC(EQ),
	OP_ASSOC(NE),
	OP_ASSOC(GT),
	OP_ASSOC(LT),
	OP_ASSOC(MIN),
	OP_ASSOC(MAX),

	OP_ASSOC(RAND_BINO),
	OP_ASSOC(RAND_UNIF),
	OP_ASSOC(RAND_NORM),

	OP_ASSOC(N_ELEMS),
	OP_ASSOC(N_DIMS),

	OP_ASSOC(ARGMAX),
	OP_ASSOC(RMAX),
	OP_ASSOC(RSUM),

	OP_ASSOC(MATMUL),
	// OP_ASSOC(CONVOLUTE),

	OP_ASSOC(PERMUTE),
	OP_ASSOC(EXTEND),
};

std::string opname (OPCODE opcode)
{
	auto it = opnames.find(opcode);
	if (opnames.end() == it)
	{
		return "BAD_OP";
	}
	return it->second;
}

#undef OP_ASSOC

}

#endif
