#include <string>

#ifndef SAND_OPCODE_HPP
#define SAND_OPCODE_HPP

enum OPCODE
{
	COPY = 0, // copy ops have no corresponding operator
	TYPECAST,
	ABS,
	NEG,
	NOT,
	SIN,
	COS,
	TAN,
	EXP,
	LOG,
	SQRT,
	ROUND,
	FLIP,
	TRANSPOSE,
	N_ELEMS,
	N_DIMS,

	POW,
	ADD,
	SUB,
	MUL,
	DIV,
	EQ,
	NE,
	LT,
	GT,
	MATMUL,
	BINO,
	UNIF,
	NORM,

	ARGMAX,
	RMAX,
	RSUM,
};

std::string opname (OPCODE opcode);

#endif /* SAND_OPCODE_HPP */
