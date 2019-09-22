#include "teq/teq.hpp"

#include "tag/prop.hpp"

#ifndef OPT_STATS_HPP
#define OPT_STATS_HPP

namespace opt
{

bool is_scalar (teq::iLeaf* leaf);

// ==== CoordptrT stringification + comparators

std::string to_string (teq::CoordptrT c);

bool lt (teq::CoordptrT a, teq::CoordptrT b);

bool is_equal (teq::CoordptrT a, teq::CoordptrT b);

// ==== Leaf comparators

bool lt (std::unordered_set<teq::iTensor*> priorities,
	teq::iLeaf* a, teq::iLeaf* b);

// for any ileaf pair a-b, they are equivalent IFF they are both tagged immutable AND
// share same shape and data values
bool is_equal (teq::iLeaf* a, teq::iLeaf* b);

// ==== Functor comparators

bool lt (std::unordered_set<teq::iTensor*> priorities,
	teq::iFunctor* a, teq::iFunctor* b);

// for any functors a-b, they are equivalent IFF a and b are the same opcode AND
// share identical function arguments (same children, shapers, and coorders)
// order matters UNLESS the op is tagged as commutative
bool is_equal (teq::iFunctor* a, teq::iFunctor* b);

}

#endif // OPT_STATS_HPP
