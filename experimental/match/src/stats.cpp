#include "experimental/match/stats.hpp"

#ifdef OPT_MATCH_STATS_HPP

namespace opt
{

namespace match
{

bool is_scalar (ade::iLeaf* leaf)
{
    ade::Shape shape = leaf->shape();
    char* data = (char*) leaf->data();
    size_t n = shape.n_elems();
    size_t perbytes = leaf->nbytes() / n;
    for (size_t i = 1; i < n; ++i)
    {
        if (false == std::equal(data, data + perbytes,
            data + i * perbytes))
        {
            return false;
        }
    }
    return true;
}

bool is_equal (ade::CoordptrT a, ade::CoordptrT b)
{
	if (a == b)
	{
		return true;
	}
	if (ade::is_identity(a.get()) && ade::is_identity(b.get()))
	{
		return true;
	}
	if (nullptr != a && nullptr != b)
	{
		return a->to_string() == b->to_string();
	}
	return false;
}

}

}

#endif
