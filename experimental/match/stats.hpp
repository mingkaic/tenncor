#include "ade/ileaf.hpp"
#include "ade/coord.hpp"

#ifndef OPT_MATCH_STATS_HPP
#define OPT_MATCH_STATS_HPP

namespace opt
{

namespace match
{

bool is_scalar (ade::iLeaf* leaf);

std::string to_string (ade::CoordptrT c);

bool lt (ade::CoordptrT a, ade::CoordptrT b);

bool is_equal (ade::CoordptrT a, ade::CoordptrT b);

}

}

#endif // OPT_MATCH_STATS_HPP
