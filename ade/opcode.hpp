/*!
 *
 *  opcode.hpp
 *  ade
 *
 *  Purpose:
 *  define enumerations for representing operations
 *
 */

#include <string>

#include "ade/shape.hpp"

#ifndef ADE_OPCODE_HPP
#define ADE_OPCODE_HPP

namespace ade
{

/*! Enumerated representation of operations */
enum OPCODE
{
	ABS = 0, //! make absolute
	NEG, //! make negative
	NOT, //! bitwise not
	SIN, //! sine
	COS, //! cosine
	TAN, //! tangent
	EXP, //! exponent
	LOG, //! natural log
	SQRT, //! square root
	ROUND, //! round value
	FLIP, //! flip values along a dimension

	POW, //! base ^ exponent
	ADD, //! a + b
	SUB, //! a - b
	MUL, //! a * b
	DIV, //! a / b
	EQ, //! a == b
	NE, //! a != b
	LT, //! a < b
	GT, //! a > b

	BINO, //! std::binomial_distribution(a, b)
	UNIF, //! std::uniform_distributon(a, b)
	NORM, //! std::normal_distribution(a, b)

	N_ELEMS, //! get n_elem of input shape as value
	N_DIMS, //! get value at specified dimension of input shape

	ARGMAX, //! get first flat index of the max value
	RMAX, //! get the max value
	RSUM, //! get the sum of all values

	MATMUL, //! matrix multiplication

	PERMUTE, /*! permute shape according to input indices. output shape take
	on input dimensions ordered by indices, and concatenated by unreferenced
	input dimensions ordered by input's original order */
	EXTEND, /*! concatenate input shape vector to input tensor's shape.
	expect value to expand into the new shape by duplicating */
	RESHAPE, /*! reshape input tensor's shape to new shape assuming the new
	shape has the same n_elems as old shape */

	GROUP, /*! In addition to tensor arguments, given an opcode and for each tensor,
	a list of indices to the tensor's shape dimension, execute the operation
	associated to that opcode for each of the arguments grouped by the shape indices. Todo: add details on how shapes are grouped */

	// todo: implement
	CONVOLUTE,

	// replace CLIP
	MIN,
	MAX,

	_NUM_OPS,
};

/*! Convert the OPCODE to string */
std::string opname (OPCODE opcode);

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

}

#endif /* ADE_OPCODE_HPP */
