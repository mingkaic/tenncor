///
/// opcode.hpp
/// age
///
/// Purpose:
/// Enumerate operators and define their function signature
///

#include <string>

#ifndef AGE_OPCODE_HPP
#define AGE_OPCODE_HPP

namespace age
{

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

	/// Given tensors a, and b, for every mapped index i in range
	/// [0:max_nelems], apply std::pow operator to elements a[i] and b[i]
	/// Only accept 2 arguments
	POW,
	/// Given tensors, for every mapped index i in range [0:max_nelems],
	/// sum all elements arg[i] for arg in tensors
	ADD,
	/// Given tensors a, and b, for every mapped index i in range
	/// [0:max_nelems], apply subtract elements a[i] and b[i]
	/// Only accept 2 arguments
	SUB,
	/// Given tensors, for every mapped index i in range [0:max_nelems],
	/// multiply all elements arg[i] for arg in tensors
	MUL,
	/// Given tensors a, and b, for every mapped index i in range
	/// [0:max_nelems], apply divide elements a[i] and b[i]
	/// Only accept 2 arguments
	DIV,
	/// Given tensors, for every mapped index i in range [0:max_nelems],
	/// take the minimum all elements arg[i]
	/// for arg in tensors
	MIN,
	/// Given tensors, for every mapped index i in range [0:max_nelems],
	/// take the maximum all elements arg[i]
	/// for arg in tensors
	MAX,

	/// Given tensors a, and b, for every mapped index i in range
	/// [0:max_nelems], apply == operator to elements a[i] and b[i]
	/// Only accept 2 arguments
	EQ,
	/// Given tensors a, and b, for every mapped index i in range
	/// [0:max_nelems], apply != operator to elements a[i] and b[i]
	/// Only accept 2 arguments
	NE,
	/// Given tensors a, and b, for every mapped index i in range
	/// [0:max_nelems], apply < operator to elements a[i] and b[i]
	/// Only accept 2 arguments
	LT,
	/// Given tensors a, and b, for every mapped index i in range
	/// [0:max_nelems], apply > operator to elements a[i] and b[i]
	/// Only accept 2 arguments
	GT,
	/// Given tensors a, and b, for every mapped index i in range
	/// [0:max_nelems], apply std::binomial_distribution function
	/// to elements a[i] and b[i]
	/// Only accept 2 arguments
	RAND_BINO,
	/// Given tensors a, and b, for every mapped index i in range
	/// [0:max_nelems], apply std::uniform_distributon function
	/// to elements a[i] and b[i]
	/// Only accept 2 arguments
	RAND_UNIF,
	/// Given tensors a, and b, for every mapped index i in range
	/// [0:max_nelems], apply std::normal_distribution function
	/// to elements a[i] and b[i]
	/// Only accept 2 arguments
	RAND_NORM,

	/// Sentinel value of bad opcodes
	_BAD_OP,
};

/// Return the string name of input OPCODE
std::string opname (OPCODE opcode);

/// Return the OPCODE of the string name
OPCODE name_op (std::string oname);

}

#endif // ADE_OPCODE_HPP
