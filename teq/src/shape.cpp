#include "teq/shape.hpp"

#ifdef TEQ_SHAPE_HPP

namespace teq
{

NElemT index (const Shape& shape, CoordT coord)
{
	for (RankT i = 0; i < rank_cap; i++)
	{
		DimT limit = shape.at(i);
		if (coord[i] >= limit)
		{
			logs::fatalf("cannot get index of bad coordinate %s for shape %s",
				fmts::to_string(coord.begin(), coord.end()).c_str(),
				shape.to_string().c_str());
		}
		// account for negative coordinates by (limit + c) % limit
		coord[i] = std::fmod(limit + coord[i], limit);
	}
	NElemT index = 0;
	for (RankT i = 1; i < rank_cap; i++)
	{
		index += coord[rank_cap - i];
		index *= shape.at(rank_cap - i - 1);
	}
	return index + coord[0];
}

CoordT coordinate (const Shape& shape, NElemT idx)
{
	if (idx >= shape.n_elems())
	{
		logs::fatalf("cannot get coordinate of index %d (>= shape %s)",
			idx, shape.to_string().c_str());
	}
	CoordT coord;
	DimT xd;
	auto it = shape.begin();
	for (RankT i = 0; i < rank_cap; ++i)
	{
		xd = idx % *(it + i);
		coord[i] = xd;
		idx = (idx - xd) / *(it + i);
	}
	return coord;
}

std::vector<DimT> narrow_shape (const Shape& sign)
{
	auto it = sign.begin(), et = sign.end();
	while (it != et && *(et - 1) <= 1)
	{
		--et;
	}
	return std::vector<DimT>(it, et);
}

}

#endif
