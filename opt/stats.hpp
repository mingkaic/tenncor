#include "ade/ade.hpp"

#include "tag/prop.hpp"

#ifndef OPT_STATS_HPP
#define OPT_STATS_HPP

namespace opt
{

bool is_scalar (ade::iLeaf* leaf);

// ==== CoordptrT stringification + comparators

std::string to_string (ade::CoordptrT c);

bool lt (ade::CoordptrT a, ade::CoordptrT b);

bool is_equal (ade::CoordptrT a, ade::CoordptrT b);

// ==== Leaf comparators

bool lt (std::unordered_set<ade::iTensor*> priorities,
	ade::iLeaf* a, ade::iLeaf* b);

// for any ileaf pair a-b, they are equivalent IFF they are both tagged immutable AND
// share same shape and data values
bool is_equal (ade::iLeaf* a, ade::iLeaf* b);

// ==== Functor comparators

bool lt (std::unordered_set<ade::iTensor*> priorities,
	ade::iFunctor* a, ade::iFunctor* b);

// for any functors a-b, they are equivalent IFF a and b are the same opcode AND
// share identical function arguments (same children, shapers, and coorders)
// order matters UNLESS the op is tagged as commutative
bool is_equal (ade::iFunctor* a, ade::iFunctor* b);

}

#endif // OPT_STATS_HPP
