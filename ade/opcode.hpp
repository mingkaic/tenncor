#include <string>

#include "ade/shape.hpp"

#ifndef ADE_OPCODE_HPP
#define ADE_OPCODE_HPP

namespace ade
{

enum OPCODE
{
	ABS = 0,
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

	POW,
	ADD,
	SUB,
	MUL,
	DIV,
	EQ,
	NE,
	LT,
	GT,

	BINO,
	UNIF,
	NORM,

	N_ELEMS,
	N_DIMS,

	ARGMAX,
	RMAX,
	RSUM,

	MATMUL,

	PERMUTE,
	EXTEND,
	RESHAPE,

	// todo: implement (replace CLIP)
	MIN,
	MAX,

	_NUM_OPS,
};

#define _DECL_NOARG(OUT, NAME, CODE, ...)\
template <> OUT NAME<CODE> (__VA_ARGS__);

#define _DECL_INTARG(OUT, NAME, CODE, ...)\
template <> OUT NAME<CODE,uint8_t> (__VA_ARGS__, uint8_t);

#define _DECL_SHPARG(OUT, NAME, CODE, ...)\
template <> OUT NAME<CODE,std::vector<DimT>> (__VA_ARGS__,std::vector<DimT>);

// define functions (template) mapped by opcode
#define _SIGNATURE_DEF(OUT, NAME, ...)\
_DECL_NOARG(OUT, NAME, ABS, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, NEG, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, NOT, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, SIN, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, COS, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, TAN, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, EXP, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, LOG, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, SQRT, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, ROUND, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, FLIP, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, POW, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, ADD, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, SUB, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, MUL, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, DIV, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, EQ, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, NE, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, LT, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, GT, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, BINO, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, UNIF, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, NORM, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, N_ELEMS, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, N_DIMS, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, ARGMAX, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, RMAX, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, RSUM, __VA_ARGS__)\
template <> OUT NAME<MATMUL> (__VA_ARGS__);\
template <> OUT NAME<MATMUL,uint8_t,uint8_t> (\
	__VA_ARGS__,uint8_t,uint8_t);\
template <> OUT NAME<PERMUTE,std::vector<uint8_t>> (\
	__VA_ARGS__,std::vector<uint8_t>);\
_DECL_SHPARG(OUT, NAME, EXTEND, __VA_ARGS__)\
_DECL_SHPARG(OUT, NAME, RESHAPE, __VA_ARGS__)

std::string opname (OPCODE opcode);

}

#endif /* ADE_OPCODE_HPP */
