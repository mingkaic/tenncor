#include "util/error.hpp"

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
			util::handle_error("invalid coordinate",
				util::ErrArg<std::vector<uint8_t>>("coord", coord),
				util::ErrArg<std::string>("shape", shape.to_string()));
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
		util::handle_error("index greater than maximum number of elements",
			util::ErrArg<size_t>("index", idx),
			util::ErrArg<std::string>("shape", shape.to_string()));
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
