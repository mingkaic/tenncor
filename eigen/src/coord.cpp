#include "eigen/coord.hpp"

#ifdef EIGEN_COORD_HPP

namespace eigen
{

CoordptrT reduce (std::set<teq::RankT> rdims)
{
	size_t n_red = rdims.size();
	if (std::any_of(rdims.begin(), rdims.end(),
		[](teq::RankT d) { return d >= teq::rank_cap; }))
	{
		logs::fatalf(
			"cannot reduce using dimensions greater or equal to rank_cap: %s",
			fmts::to_string(rdims.begin(), rdims.end()).c_str());
	}
	if (n_red > teq::rank_cap)
	{
		logs::fatalf("cannot reduce %d rank when only ranks are capped at %d",
			n_red, teq::rank_cap);
	}

	teq::CoordT dims;
	auto it = dims.begin();
	std::fill(it, dims.end(), teq::rank_cap);
	std::copy(rdims.begin(), rdims.end(), it);
	return std::make_shared<CoordMap>(dims);
}

CoordptrT extend (teq::CoordT bcast)
{
	if (std::any_of(bcast.begin(), bcast.end(),
		[](teq::RankT rank) { return rank < 1; }))
	{
		logs::fatalf("cannot extend with zero values: %s",
			fmts::to_string(bcast.begin(), bcast.end()).c_str());
	}

	return std::make_shared<CoordMap>(bcast);
}

CoordptrT permute (std::array<teq::RankT,teq::rank_cap> order)
{
	teq::CoordT dims;
	std::copy(order.begin(), order.end(), dims.begin());
	return std::make_shared<CoordMap>(dims);
}

}

#endif
