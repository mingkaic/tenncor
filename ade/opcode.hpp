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

// TODO: CONVERT TO GENERATED CONFIG

/// Enumerated representation of operations
enum OPCODE
{
	COPY = 0,

	/// Make every element positive
	ABS,
	/// Negate the sign of every element
	NEG,
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

	_BAD_OP,
};

/// Return the string name of input OPCODE
std::string opname (OPCODE opcode);

/// Return the OPCODE of the string name
OPCODE name_op (std::string oname);

}

#endif // ADE_OPCODE_HPP
