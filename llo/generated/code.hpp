#include <string>

#ifndef _GENERATED_CODES_HPP
#define _GENERATED_CODES_HPP

namespace age
{

enum _GENERATED_OPCODE
{
	BAD_OP = 0,
	COS,
	LOG,
	MIN,
	NEG,
	SUM,
	EXP,
	DIV,
	ROUND,
	NEQ,
	RAND_NORM,
	POW,
	SIN,
	LT,
	TAN,
	GT,
	ABS,
	SUB,
	EQ,
	MAX,
	SQRT,
	RAND_UNIF,
	PROD,
	RAND_BINO,
};

enum _GENERATED_DTYPE
{
	BAD_TYPE = 0,
	UINT64,
	INT32,
	INT16,
	DOUBLE,
	FLOAT,
	UINT8,
	UINT32,
	UINT16,
	INT64,
	INT8,
};

std::string name_op (_GENERATED_OPCODE code);

_GENERATED_OPCODE get_op (std::string name);

std::string name_type (_GENERATED_DTYPE type);

uint8_t type_size (_GENERATED_DTYPE type);

_GENERATED_DTYPE get_type (std::string name);

template <typename T>
_GENERATED_DTYPE get_type (void)
{
	return BAD_TYPE;
}

template <>
_GENERATED_DTYPE get_type<uint64_t> (void);

template <>
_GENERATED_DTYPE get_type<int32_t> (void);

template <>
_GENERATED_DTYPE get_type<int16_t> (void);

template <>
_GENERATED_DTYPE get_type<double> (void);

template <>
_GENERATED_DTYPE get_type<float> (void);

template <>
_GENERATED_DTYPE get_type<uint8_t> (void);

template <>
_GENERATED_DTYPE get_type<uint32_t> (void);

template <>
_GENERATED_DTYPE get_type<uint16_t> (void);

template <>
_GENERATED_DTYPE get_type<int64_t> (void);

template <>
_GENERATED_DTYPE get_type<int8_t> (void);

}

#endif // _GENERATED_CODES_HPP
