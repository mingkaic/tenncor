#include <unordered_map>
#include "err/log.hpp"
#include "llo/generated/code.hpp"

#ifdef _GENERATED_CODES_HPP

namespace age
{

struct EnumHash
{
	template <typename T>
	size_t operator() (T e) const
	{
		return static_cast<size_t>(e);
	}
};

static std::unordered_map<_GENERATED_OPCODE,std::string,EnumHash> code2name =
{
	{ COS, "COS" },
	{ LOG, "LOG" },
	{ MIN, "MIN" },
	{ NEG, "NEG" },
	{ SUM, "SUM" },
	{ EXP, "EXP" },
	{ DIV, "DIV" },
	{ ROUND, "ROUND" },
	{ NEQ, "NEQ" },
	{ RAND_NORM, "RAND_NORM" },
	{ POW, "POW" },
	{ SIN, "SIN" },
	{ LT, "LT" },
	{ TAN, "TAN" },
	{ GT, "GT" },
	{ ABS, "ABS" },
	{ SUB, "SUB" },
	{ EQ, "EQ" },
	{ MAX, "MAX" },
	{ SQRT, "SQRT" },
	{ RAND_UNIF, "RAND_UNIF" },
	{ PROD, "PROD" },
	{ RAND_BINO, "RAND_BINO" },
};

static std::unordered_map<std::string,_GENERATED_OPCODE> name2code =
{
	{ "COS", COS },
	{ "LOG", LOG },
	{ "MIN", MIN },
	{ "NEG", NEG },
	{ "SUM", SUM },
	{ "EXP", EXP },
	{ "DIV", DIV },
	{ "ROUND", ROUND },
	{ "NEQ", NEQ },
	{ "RAND_NORM", RAND_NORM },
	{ "POW", POW },
	{ "SIN", SIN },
	{ "LT", LT },
	{ "TAN", TAN },
	{ "GT", GT },
	{ "ABS", ABS },
	{ "SUB", SUB },
	{ "EQ", EQ },
	{ "MAX", MAX },
	{ "SQRT", SQRT },
	{ "RAND_UNIF", RAND_UNIF },
	{ "PROD", PROD },
	{ "RAND_BINO", RAND_BINO },
};

const std::unordered_map<_GENERATED_DTYPE,std::string,EnumHash> type2name =
{
	{ UINT64, "UINT64" },
	{ INT32, "INT32" },
	{ INT16, "INT16" },
	{ DOUBLE, "DOUBLE" },
	{ FLOAT, "FLOAT" },
	{ UINT8, "UINT8" },
	{ UINT32, "UINT32" },
	{ UINT16, "UINT16" },
	{ INT64, "INT64" },
	{ INT8, "INT8" },
};

static std::unordered_map<std::string,_GENERATED_DTYPE> name2type =
{
	{ "UINT64", UINT64 },
	{ "INT32", INT32 },
	{ "INT16", INT16 },
	{ "DOUBLE", DOUBLE },
	{ "FLOAT", FLOAT },
	{ "UINT8", UINT8 },
	{ "UINT32", UINT32 },
	{ "UINT16", UINT16 },
	{ "INT64", INT64 },
	{ "INT8", INT8 },
};

std::string name_op (_GENERATED_OPCODE code)
{
	auto it = code2name.find(code);
	if (code2name.end() == it)
	{
		return "BAD_OP";
	}
	return it->second;
}

_GENERATED_OPCODE get_op (std::string name)
{
	auto it = name2code.find(name);
	if (name2code.end() == it)
	{
		return BAD_OP;
	}
	return it->second;
}

std::string name_type (_GENERATED_DTYPE type)
{
	auto it = type2name.find(type);
	if (type2name.end() == it)
	{
		return "BAD_TYPE";
	}
	return it->second;
}

_GENERATED_DTYPE get_type (std::string name)
{
	auto it = name2type.find(name);
	if (name2type.end() == it)
	{
		return BAD_TYPE;
	}
	return it->second;
}

uint8_t type_size (_GENERATED_DTYPE type)
{
	switch (type)
	{
		case UINT64: return sizeof(uint64_t);
		case INT32: return sizeof(int32_t);
		case INT16: return sizeof(int16_t);
		case DOUBLE: return sizeof(double);
		case FLOAT: return sizeof(float);
		case UINT8: return sizeof(uint8_t);
		case UINT32: return sizeof(uint32_t);
		case UINT16: return sizeof(uint16_t);
		case INT64: return sizeof(int64_t);
		case INT8: return sizeof(int8_t);
		default: err::fatal("cannot get size of bad type");
	}
}

template <>
_GENERATED_DTYPE get_type<uint64_t> (void)
{
	return UINT64;
}

template <>
_GENERATED_DTYPE get_type<int32_t> (void)
{
	return INT32;
}

template <>
_GENERATED_DTYPE get_type<int16_t> (void)
{
	return INT16;
}

template <>
_GENERATED_DTYPE get_type<double> (void)
{
	return DOUBLE;
}

template <>
_GENERATED_DTYPE get_type<float> (void)
{
	return FLOAT;
}

template <>
_GENERATED_DTYPE get_type<uint8_t> (void)
{
	return UINT8;
}

template <>
_GENERATED_DTYPE get_type<uint32_t> (void)
{
	return UINT32;
}

template <>
_GENERATED_DTYPE get_type<uint16_t> (void)
{
	return UINT16;
}

template <>
_GENERATED_DTYPE get_type<int64_t> (void)
{
	return INT64;
}

template <>
_GENERATED_DTYPE get_type<int8_t> (void)
{
	return INT8;
}

}

#endif
