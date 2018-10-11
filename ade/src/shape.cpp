#include "ade/log.hpp"

#include "ade/shape.hpp"

#ifdef ADE_SHAPE_HPP

namespace ade
{

NElemT index (Shape shape, std::vector<DimT> coord)
{
	uint8_t n = std::min((size_t) shape.n_rank(), coord.size());
	for (uint8_t i = 0; i < n; i++)
	{
		if (coord[i] >= shape.at(i))
		{
			fatalf("cannot get index of bad coordinate %s for shape %s",
				to_string(coord), shape.to_string());
		}
	}
	NElemT index = 0;
	for (uint8_t i = 1; i < n; i++)
	{
		index += coord[n - i];
		index *= shape.at(n - i - 1);
	}
	return index + coord[0];
}

std::vector<DimT> coordinate (Shape shape, NElemT idx)
{
	if (idx >= shape.n_elems())
	{
		fatalf("cannot get coordinate of index %d (>= shape %s nelems)",
			idx, shape.to_string());
	}
	std::vector<DimT> coord;
	DimT xd;
	auto it = shape.begin();
	for (auto et = it + shape.n_rank(); it != et; ++it)
	{
		xd = idx % *it;
		coord.push_back(xd);
		idx = (idx - xd) / *it;
	}
	return coord;
}

}

#endif
