#include "tenncor/layr/layer.hpp"

#ifdef LAYR_LAYER_HPP

namespace layr
{

teq::Shape gen_rshape (std::vector<teq::DimT> runcoms,
	teq::Shape left, eigen::PairVecT<teq::RankT> lrdims)
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

teq::TensptrT make_layer (teq::TensptrT root,
	const std::string& layername, teq::TensptrT input)
{
	auto f = estd::must_cast<teq::iFunctor>(root.get());
	if (nullptr != f->get_attr(teq::layer_key))
	{
		global::fatalf("attempting to attach layer attribute to node %s "
			"with an existing layer attribute", root->to_string().c_str());
	}
	f->add_attr(teq::layer_key,
		std::make_unique<teq::LayerObj>(layername, input));
	return root;
}

}

#endif
