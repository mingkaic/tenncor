#include "layr/api.hpp"

#ifdef LAYR_API_HPP

namespace layr
{

teq::Shape gen_rshape (std::vector<teq::DimT> runcoms,
	teq::ShapeSignature left, eigen::PairVecT<teq::RankT> lrdims)
{
	// split runcoms values that avoids right dimensions in lrdims
	std::array<bool,teq::rank_cap> unvisited;
	std::vector<teq::DimT> slist(teq::rank_cap, 1);
	std::fill(unvisited.begin(), unvisited.end(), true);
	for (auto& lr : lrdims)
	{
		slist[lr.second] = left.at(lr.first);
		unvisited[lr.second] = false;
	}
	for (size_t i = 0, j = 0, n = runcoms.size();
		i < teq::rank_cap && j < n; ++i)
	{
		if (unvisited[i])
		{
			slist[i] = runcoms[j];
			++j;
		}
	}
	return teq::Shape(slist);
}

}

#endif
