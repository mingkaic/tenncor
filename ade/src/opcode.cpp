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

#define CODE_ASSOC(CODE) std::pair<OPCODE,std::string>{CODE, #CODE}
#define NAME_ASSOC(CODE) std::pair<std::string,OPCODE>{#CODE, CODE}

const std::unordered_map<OPCODE,std::string,EnumHash> opnames =
{
	CODE_ASSOC(COPY),
	CODE_ASSOC(ABS),
	CODE_ASSOC(NEG),
	CODE_ASSOC(SIN),
	CODE_ASSOC(COS),
	CODE_ASSOC(TAN),
	CODE_ASSOC(EXP),
	CODE_ASSOC(LOG),
	CODE_ASSOC(SQRT),
	CODE_ASSOC(ROUND),

	CODE_ASSOC(POW),
	CODE_ASSOC(ADD),
	CODE_ASSOC(SUB),
	CODE_ASSOC(MUL),
	CODE_ASSOC(DIV),
	CODE_ASSOC(EQ),
	CODE_ASSOC(NE),
	CODE_ASSOC(GT),
	CODE_ASSOC(LT),
	CODE_ASSOC(MIN),
	CODE_ASSOC(MAX),

	CODE_ASSOC(RAND_BINO),
	CODE_ASSOC(RAND_UNIF),
	CODE_ASSOC(RAND_NORM),
};

const std::unordered_map<std::string,OPCODE> opcodes =
{
	NAME_ASSOC(COPY),
	NAME_ASSOC(ABS),
	NAME_ASSOC(NEG),
	NAME_ASSOC(SIN),
	NAME_ASSOC(COS),
	NAME_ASSOC(TAN),
	NAME_ASSOC(EXP),
	NAME_ASSOC(LOG),
	NAME_ASSOC(SQRT),
	NAME_ASSOC(ROUND),

	NAME_ASSOC(POW),
	NAME_ASSOC(ADD),
	NAME_ASSOC(SUB),
	NAME_ASSOC(MUL),
	NAME_ASSOC(DIV),
	NAME_ASSOC(EQ),
	NAME_ASSOC(NE),
	NAME_ASSOC(GT),
	NAME_ASSOC(LT),
	NAME_ASSOC(MIN),
	NAME_ASSOC(MAX),

	NAME_ASSOC(RAND_BINO),
	NAME_ASSOC(RAND_UNIF),
	NAME_ASSOC(RAND_NORM),
};

#undef CODE_ASSOC
#undef NAME_ASSOC

std::string opname (OPCODE opcode)
{
	auto it = opnames.find(opcode);
	if (opnames.end() == it)
	{
		return "BAD_OP";
	}
	return it->second;
}

OPCODE name_op (std::string oname)
{
	auto it = opcodes.find(oname);
	if (opcodes.end() == it)
	{
		return _BAD_OP;
	}
	return it->second;
}

}

#endif
