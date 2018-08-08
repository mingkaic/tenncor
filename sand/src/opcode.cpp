#include "sand/opcode.hpp"
#include "util/mapper.hpp"

#ifdef SAND_OPCODE_HPP

#define OP_ASSOC(CODE) std::pair<OPCODE,std::string>{CODE, #CODE}

using OpnameMap = EnumMap<OPCODE,std::string>;

const OpnameMap opnames =
{
	OP_ASSOC(TYPECAST),
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
	OP_ASSOC(TRANSPOSE),
	OP_ASSOC(N_ELEMS),
	OP_ASSOC(N_DIMS),

	OP_ASSOC(POW),
	OP_ASSOC(ADD),
	OP_ASSOC(SUB),
	OP_ASSOC(MUL),
	OP_ASSOC(DIV),
	OP_ASSOC(EQ),
	OP_ASSOC(NE),
	OP_ASSOC(GT),
	OP_ASSOC(LT),
	OP_ASSOC(MATMUL),

	OP_ASSOC(BINO),
	OP_ASSOC(UNIF),
	OP_ASSOC(NORM),

	OP_ASSOC(ARGMAX),
	OP_ASSOC(RMAX),
	OP_ASSOC(RSUM),
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

#endif
