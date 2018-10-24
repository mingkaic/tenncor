#include "ade/log.hpp"

#include "ade/shape.hpp"

#ifdef ADE_SHAPE_HPP

namespace ade
{

NElemT index (Shape shape, CoordT coord)
{
	for (uint8_t i = 0; i < rank_cap; i++)
	{
		if (coord[i] >= shape.at(i))
		{
			fatalf("cannot get index of bad coordinate %s for shape %s",
				to_string(coord).c_str(), shape.to_string().c_str());
		}
	}
	NElemT index = 0;
	for (uint8_t i = 1; i < rank_cap; i++)
	{
		index += coord[rank_cap - i];
		index *= shape.at(rank_cap - i - 1);
	}
	return index + coord[0];
}

CoordT coordinate (Shape shape, NElemT idx)
{
	if (idx >= shape.n_elems())
	{
		fatalf("cannot get coordinate of index %d (>= shape %s nelems)",
			idx, shape.to_string().c_str());
	}
	CoordT coord;
	DimT xd;
	auto it = shape.begin();
	for (uint8_t i = 0; i < rank_cap; ++i)
	{
		xd = idx % *(it + i);
		coord[i] = xd;
		idx = (idx - xd) / *(it + i);
	}
	return coord;
}

}

#endif
