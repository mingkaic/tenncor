///
/// opcode.hpp
/// ade
///
/// Purpose:
/// Enumerate operators and define their function signature
///

#include <string>

#include "ade/shape.hpp"

#ifndef ADE_OPCODE_HPP
#define ADE_OPCODE_HPP

namespace ade
{

/// Enumerated representation of operations
enum OPCODE
{
	/// Make every element positive
	ABS = 0,
	/// Negate the sign of every element
	NEG,
	/// Bitwise not every element
	NOT,
	/// Sine every element
	SIN,
	/// Cosine every element
	COS,
	/// Tangent every element
	TAN,
	/// Exponent every element
	EXP,
	/// Natural log every element
	LOG,
	/// Square root every element
	SQRT,
	/// Round every element
	ROUND,
	/// Flip element along a specific dimension
	/// For example, given 2-D tensor [[1, 2], [3, 4]], dim=1,
	/// output tensor is [[3, 4], [1, 2]]
	FLIP,

	/// Given tensors a, and b, for every index i in range [0:max_nelems],
	/// apply std::pow operator to elements a[i % a.nelems] and b[i % b.nelems]
	/// Shapes must be compatible before min_rank of both tensors
	/// Only accept 2 arguments
	POW,
	/// Given tensors, for every index i in range [0:max_nelems],
	/// sum all elements arg[i % arg.nelems] for arg in tensors
	/// Shapes must be compatible before min_rank of all tensors
	ADD,
	/// Given tensors a, and b, for every index i in range [0:max_nelems],
	/// apply subtract elements a[i % a.nelems] and b[i % b.nelems]
	/// Shapes must be compatible before min_rank of both tensors
	/// Only accept 2 arguments
	SUB,
	/// Given tensors, for every index i in range [0:max_nelems],
	/// multiply all elements arg[i % arg.nelems] for arg in tensors
	/// Shapes must be compatible before min_rank of all tensors
	MUL,
	/// Given tensors a, and b, for every index i in range [0:max_nelems],
	/// apply divide elements a[i % a.nelems] and b[i % b.nelems]
	/// Shapes must be compatible before min_rank of both tensors
	/// Only accept 2 arguments
	DIV,
	/// Given tensors a, and b, for every index i in range [0:max_nelems],
	/// apply == operator to elements a[i % a.nelems] and b[i % b.nelems]
	/// Shapes must be compatible before min_rank of both tensors
	/// Only accept 2 arguments
	EQ,
	/// Given tensors a, and b, for every index i in range [0:max_nelems],
	/// apply != operator to elements a[i % a.nelems] and b[i % b.nelems]
	/// Shapes must be compatible before min_rank of both tensors
	/// Only accept 2 arguments
	NE,
	/// Given tensors a, and b, for every index i in range [0:max_nelems],
	/// apply < operator to elements a[i % a.nelems] and b[i % b.nelems]
	/// Shapes must be compatible before min_rank of both tensors
	/// Only accept 2 arguments
	LT,
	/// Given tensors a, and b, for every index i in range [0:max_nelems],
	/// apply > operator to elements a[i % a.nelems] and b[i % b.nelems]
	/// Shapes must be compatible before min_rank of both tensors
	/// Only accept 2 arguments
	GT,
	/// Given tensors, for every index i in range [0:max_nelems],
	/// take the minimum all elements arg[i % arg.nelems]
	/// for arg in tensors
	/// Shapes must be compatible before min_rank of all tensors
	MIN,
	/// Given tensors, for every index i in range [0:max_nelems],
	/// take the maximum all elements arg[i % arg.nelems]
	/// for arg in tensors
	/// Shapes must be compatible before min_rank of all tensors
	MAX,

	/// Given tensors a, and b, for every index i in range [0:max_nelems],
	/// apply std::binomial_distribution function
	/// to elements a[i % a.nelems] and b[i % b.nelems]
	/// Shapes must be compatible before min_rank of both tensors
	/// Only accept 2 arguments
	RAND_BINO,
	/// Given tensors a, and b, for every index i in range [0:max_nelems],
	/// apply std::uniform_distributon function
	/// to elements a[i % a.nelems] and b[i % b.nelems]
	/// Shapes must be compatible before min_rank of both tensors
	/// Only accept 2 arguments
	RAND_UNIF,
	/// Given tensors a, and b, for every index i in range [0:max_nelems],
	/// apply std::normal_distribution function
	/// to elements a[i % a.nelems] and b[i % b.nelems]
	/// Shapes must be compatible before min_rank of both tensors
	/// Only accept 2 arguments
	RAND_NORM,

	/// Given a single argument get the n_elem value of the argument's shape
	N_ELEMS,
	/// Given an argument and a dimension index get the value of the argument's
	/// shape at that index
	N_DIMS,

	/// Out of all elements, get first flat index of the max value
	ARGMAX,
	/// Out of all elements in an argument, get the max value
	RMAX,
	/// Out of all elements in an argument, get the sum of all values
	RSUM,

	/// Given 2 tensors, matrix multiply
	/// The # of column of the first argument must match the nrow of the second
	/// Given the tensors and 2 indices, for each argument
	/// form groups [:idx) and [index:rank) and treat dimensions falling in
	/// those ranges as a single dimension (where the shape values must match)
	/// then apply matmul given the grouped shape
	/// For example, given shapea={3, 4, 5}, ai=2, shapeb={7, 8, 3, 4}, bi=2,
	/// output tensor has shape {7, 8, 5}, since {3, 4} in a and b matches
	MATMUL,

	/// <<< UNIMPLEMENTED >>>
	CONVOLUTE,

	/// Given a tensor argument and a vector of shape indices, permute tensor
	/// The tensor's shape and each element's coordinates are reordered
	/// according to shape indices
	/// Unreferenced input shape dimensions are appended to the output shape
	/// Input dimensions can be referenced more than once
	/// Because output.nelems >= input.nelems, and coordinates are mapped 1-1,
	/// Output indices that do not take input elements take on value 0
	/// This 1-1 behavior is to facilitate creating identity matrices/tensors
	PERMUTE,
	/// Given a tensor and a vector of dimension values, append dimension
	/// values to tensor shape so that output.nelems >= input.nelems
	/// tensor data duplicate to accommodate new shape
	/// Combined with PERMUTE, EXTEND can remap values 1-N
	EXTEND,
	/// Given a tensor and a shape, output tensor's data with new shape without
	/// changing the coordinate of each element
	/// Report error if new shape.nelems != old shape.nelems
	RESHAPE,

	_NUM_OPS,
};

/// Return the string name of input OPCODE
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
_DECL_NOARG(OUT, NAME, MIN, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, MAX, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, RAND_BINO, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, RAND_UNIF, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, RAND_NORM, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, N_ELEMS, __VA_ARGS__)\
_DECL_NOARG(OUT, NAME, N_DIMS, __VA_ARGS__)\
_DECL_INTARG(OUT, NAME, ARGMAX, __VA_ARGS__)\
_DECL_INTARG(OUT, NAME, RMAX, __VA_ARGS__)\
_DECL_INTARG(OUT, NAME, RSUM, __VA_ARGS__)\
template <> OUT NAME<MATMUL,uint8_t,uint8_t> (\
	__VA_ARGS__,uint8_t,uint8_t);\
template <> OUT NAME<PERMUTE,std::vector<uint8_t>> (\
	__VA_ARGS__,std::vector<uint8_t>);\
_DECL_SHPARG(OUT, NAME, EXTEND, __VA_ARGS__)\
_DECL_SHPARG(OUT, NAME, RESHAPE, __VA_ARGS__)

}

#endif // ADE_OPCODE_HPP
