#include "ead/coord.hpp"

#ifdef EAD_COORD_HPP

namespace ead
{

ade::CoordptrT reduce (uint8_t rank, std::vector<uint8_t> red)
{
	uint8_t n_red = red.size();
	if (std::any_of(red.begin(), red.end(),
		[](ade::DimT& d) { return d >= ade::rank_cap; }))
	{
		logs::fatalf(
			"cannot reduce using dimensions greater or equal to rank_cap: %s",
			fmts::to_string(red.begin(), red.end()).c_str());
	}
	if (rank + n_red > ade::rank_cap)
	{
		logs::fatalf("cannot reduce shape rank %d beyond rank_cap with n_red %d",
			rank, n_red);
	}
	if (0 == n_red)
	{
		logs::warn("reducing with empty vector ... will do nothing");
		return nullptr;
	}

	ade::CoordT rdims;
	auto it = rdims.begin();
	std::fill(it, rdims.end(), ade::rank_cap);
	std::copy(red.begin(), red.end(), it + rank);
	return ade::CoordptrT(new CoordMap(rdims, false));
}

ade::CoordptrT extend (uint8_t rank, std::vector<ade::DimT> ext)
{
	uint8_t n_ext = ext.size();
	if (std::any_of(ext.begin(), ext.end(),
		[](ade::DimT& d) { return 0 == d; }))
	{
		logs::fatalf("cannot extend using zero dimensions %s",
			fmts::to_string(ext.begin(), ext.end()).c_str());
	}
	if (rank + n_ext > ade::rank_cap)
	{
		logs::fatalf("cannot extend shape rank %d beyond rank_cap with n_ext %d",
			rank, n_ext);
	}
	if (0 == n_ext)
	{
		logs::warn("extending with empty vector ... will do nothing");
		return nullptr;
	}

	ade::CoordT bcast;
	auto it = bcast.begin();
	std::fill(it, bcast.end(), 1);
	std::copy(ext.begin(), ext.end(), it + rank);
	return ade::CoordptrT(new CoordMap(bcast, false));
}

ade::CoordptrT permute (std::vector<uint8_t> dims)
{
	if (dims.size() == 0)
	{
		logs::warn("permuting with same dimensions ... will do nothing");
		return nullptr;
	}

	bool visited[ade::rank_cap];
	std::memset(visited, false, ade::rank_cap);
	for (uint8_t i = 0, n = dims.size(); i < n; ++i)
	{
		if (visited[dims[i]])
		{
			logs::fatalf("permute does not support repeated orders: %s",
				fmts::to_string(dims.begin(), dims.end()).c_str());
		}
		visited[dims[i]] = true;
	}
	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		if (false == visited[i])
		{
			dims.push_back(i);
		}
	}

	ade::CoordT order;
	std::copy(dims.begin(), dims.end(), order.begin());
	return ade::CoordptrT(new CoordMap(order, true));
}

}

#endif
