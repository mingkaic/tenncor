#include "ead/coord.hpp"

#ifdef EAD_COORD_HPP

namespace ead
{

CoordptrT reduce (std::vector<ade::RankT> red_dims)
{
	ade::RankT n_red = red_dims.size();
	if (std::any_of(red_dims.begin(), red_dims.end(),
		[](ade::RankT& d) { return d >= ade::rank_cap; }))
	{
		logs::fatalf(
			"cannot reduce using dimensions greater or equal to rank_cap: %s",
			fmts::to_string(red_dims.begin(), red_dims.end()).c_str());
	}
	if (n_red > ade::rank_cap)
	{
		logs::fatalf("cannot reduce %d rank when only ranks are capped at %d",
			n_red, ade::rank_cap);
	}

	ade::CoordT rdims;
	auto it = rdims.begin();
	std::fill(it, rdims.end(), ade::rank_cap);
	std::copy(red_dims.begin(), red_dims.end(), it);
	return std::make_shared<CoordMap>(rdims, false);
}

CoordptrT extend (ade::RankT rank, std::vector<ade::DimT> ext)
{
	ade::RankT n_ext = ext.size();
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
	return std::make_shared<CoordMap>(bcast, false);
}

CoordptrT permute (std::vector<ade::RankT> dims)
{
	if (dims.size() == 0)
	{
		logs::warn("permuting with same dimensions ... will do nothing");
		return nullptr;
	}

	bool visited[ade::rank_cap];
	std::memset(visited, false, ade::rank_cap);
	for (ade::RankT i = 0, n = dims.size(); i < n; ++i)
	{
		if (visited[dims[i]])
		{
			logs::fatalf("permute does not support repeated orders: %s",
				fmts::to_string(dims.begin(), dims.end()).c_str());
		}
		visited[dims[i]] = true;
	}
	for (ade::RankT i = 0; i < ade::rank_cap; ++i)
	{
		if (false == visited[i])
		{
			dims.push_back(i);
		}
	}

	ade::CoordT order;
	std::copy(dims.begin(), dims.end(), order.begin());
	return std::make_shared<CoordMap>(order, true);
}

}

#endif
