#include "ade/ade.hpp"

#include "tag/prop.hpp"

#ifndef OPT_COMPARE_HPP
#define OPT_COMPARE_HPP

namespace opt
{

bool lt (ade::CoordptrT a, ade::CoordptrT b);

bool lt (ade::FuncArg a, ade::FuncArg b,
	std::function<bool(const ade::TensptrT&,const ade::TensptrT&)> teneq =
	[](const ade::TensptrT& a, const ade::TensptrT& b)
	{
		return a.get() == b.get();
	},
	std::function<bool(const ade::TensptrT&,const ade::TensptrT&)> tencmp =
	[](const ade::TensptrT& a, const ade::TensptrT& b)
	{
		return a.get() < b.get();
	});

bool lt (std::unordered_set<ade::iTensor*> priorities,
	ade::iLeaf* a, ade::iLeaf* b);

bool lt (std::unordered_set<ade::iTensor*> priorities,
	ade::iFunctor* a, ade::iFunctor* b);

bool is_equal (ade::CoordptrT a, ade::CoordptrT b);

bool is_equal (ade::FuncArg a, ade::FuncArg b);

// for any ileaf pair a-b, they are equivalent IFF they are both tagged immutable AND
// share same shape and data values
bool is_equal (ade::iLeaf* a, ade::iLeaf* b);

// for any functors a-b, they are equivalent IFF a and b are the same opcode AND
// share identical function arguments (same children, shapers, and coorders)
// order matters UNLESS the op is tagged as commutative
bool is_equal (ade::iFunctor* a, ade::iFunctor* b);

}

#endif // OPT_COMPARE_HPP
