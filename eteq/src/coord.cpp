#include "eteq/coord.hpp"

#ifdef ETEQ_COORD_HPP

namespace eteq
{

CoordptrT reduce (std::vector<teq::RankT> red_dims)
{
	teq::RankT n_red = red_dims.size();
	if (std::any_of(red_dims.begin(), red_dims.end(),
		[](teq::RankT& d) { return d >= teq::rank_cap; }))
	{
		logs::fatalf(
			"cannot reduce using dimensions greater or equal to rank_cap: %s",
			fmts::to_string(red_dims.begin(), red_dims.end()).c_str());
	}
	if (n_red > teq::rank_cap)
	{
		logs::fatalf("cannot reduce %d rank when only ranks are capped at %d",
			n_red, teq::rank_cap);
	}

	teq::CoordT rdims;
	auto it = rdims.begin();
	std::fill(it, rdims.end(), teq::rank_cap);
	std::copy(red_dims.begin(), red_dims.end(), it);
	return std::make_shared<CoordMap>(rdims);
}

CoordptrT extend (teq::RankT rank, std::vector<teq::DimT> ext)
{
	teq::RankT n_ext = ext.size();
	if (std::any_of(ext.begin(), ext.end(),
		[](teq::DimT& d) { return 0 == d; }))
	{
		logs::fatalf("cannot extend using zero dimensions %s",
			fmts::to_string(ext.begin(), ext.end()).c_str());
	}
	if (rank + n_ext > teq::rank_cap)
	{
		logs::fatalf("cannot extend shape rank %d beyond rank_cap with n_ext %d",
			rank, n_ext);
	}
	if (0 == n_ext)
	{
		logs::warn("extending with empty vector ... will do nothing");
		return nullptr;
	}

	teq::CoordT bcast;
	auto it = bcast.begin();
	std::fill(it, bcast.end(), 1);
	std::copy(ext.begin(), ext.end(), it + rank);
	return std::make_shared<CoordMap>(bcast);
}

CoordptrT permute (std::vector<teq::RankT> dims)
{
	if (dims.size() == 0)
	{
		logs::warn("permuting with same dimensions ... will do nothing");
		return nullptr;
	}

	bool visited[teq::rank_cap];
	std::fill(visited, visited + teq::rank_cap, false);
	for (teq::RankT i = 0, n = dims.size(); i < n; ++i)
	{
		if (visited[dims[i]])
		{
			logs::fatalf("permute does not support repeated orders: %s",
				fmts::to_string(dims.begin(), dims.end()).c_str());
		}
		visited[dims[i]] = true;
	}
	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		if (false == visited[i])
		{
			dims.push_back(i);
		}
	}

	teq::CoordT order;
	std::copy(dims.begin(), dims.end(), order.begin());
	return std::make_shared<CoordMap>(order, true);
}

}

#endif
