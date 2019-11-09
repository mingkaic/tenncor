///
/// stats.hpp
/// opt
///
/// Purpose:
/// Define comparator and conversion for TEQ objects
/// for OPT normalization and logging
///

#include "tag/prop.hpp"

#ifndef OPT_STATS_HPP
#define OPT_STATS_HPP

namespace opt
{

/// Return true if leaf contains a scalar
bool is_scalar (teq::iLeaf* leaf);

// ==== ShaperT stringification + comparators

/// Return brief hashable string representation of coordinate mapper
std::string to_string (teq::ShaperT c);

/// Return true if a < b according to some internal ordinal rule
bool lt (teq::ShaperT a, teq::ShaperT b);

/// Return true if a is equal to b
bool is_equal (teq::ShaperT a, teq::ShaperT b);

// ==== Leaf comparators

/// Return true if a.type_code < b.type_code or a.shape < b.shape
/// if all above properties are equal,
/// return true if a is in priorities otherwise false
bool lt (teq::TensSetT priorities,
	teq::iLeaf* a, teq::iLeaf* b);

/// Return true if a and b are both tagged immutable and
/// share the same shape and data values
bool is_equal (teq::iLeaf* a, teq::iLeaf* b);

// ==== Functor comparators

/// Return true if a.opcode < b.opcode or
/// a.shape < b.shape or a.nchildren < b.nchildren
/// if all above properties are equal,
/// return true if a is in priorities otherwise false
bool lt (teq::TensSetT priorities,
	teq::iFunctor* a, teq::iFunctor* b);

/// Return true if a and b are the same opcode and
/// have identical function arguments (same children, shapers, and coorders)
/// Argument order matters unless the op is tagged as commutative
bool is_equal (teq::iFunctor* a, teq::iFunctor* b);

}

#endif // OPT_STATS_HPP
