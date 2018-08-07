#include <string>

#ifndef SAND_OPCODE_HPP
#define SAND_OPCODE_HPP

enum OPCODE
{
	CAST = 0,
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
	ISMAX,
	FLIP,
	EXPAND,
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
	GT,
	LT,
	BINO,
	UNIF,
	NORM,
	ARGMAX,
	RMAX,
	RSUM,
	MATMUL,
	RESHAPE,
};

std::string opname (OPCODE opcode);

#endif /* SAND_OPCODE_HPP */
