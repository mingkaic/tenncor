#include "ade/ileaf.hpp"
#include "ade/coord.hpp"

#ifndef OPT_MATCH_STATS_HPP
#define OPT_MATCH_STATS_HPP

namespace opt
{

namespace match
{

bool is_scalar (ade::iLeaf* leaf);

bool lt (ade::CoordptrT a, ade::CoordptrT b)
{
	if (ade::is_identity(a.get()))
	{
		return false == ade::is_identity(b.get());
	}
	return a->to_string() < b->to_string();
}

bool is_equal (ade::CoordptrT a, ade::CoordptrT b);

std::string to_string (ade::CoordptrT c)
{
    if (ade::is_identity(c))
    {
        return "";
    }
    return c->to_string();
}

}

}

#endif // OPT_MATCH_STATS_HPP
