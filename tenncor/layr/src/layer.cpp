#include "tenncor/layr/layer.hpp"

#ifdef LAYR_LAYER_HPP

namespace layr
{

teq::Shape gen_rshape (teq::DimsT runcoms,
	teq::Shape left, eigen::PairVecT<teq::RankT> lrdims)
{
	// split runcoms values that avoids right dimensions in lrdims
	std::array<bool,teq::rank_cap> unvisited;
	teq::DimsT slist(teq::rank_cap, 1);
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
	if (nullptr != f->get_attr(teq::layer_attr))
	{
		global::fatalf("attempting to attach layer attribute to node %s "
			"with an existing layer attribute", root->to_string().c_str());
	}
	f->add_attr(teq::layer_attr,
		std::make_unique<teq::LayerObj>(layername, input));
	return root;
}

eteq::ETensor get_input (const eteq::ETensor& root)
{
	if (nullptr == root)
	{
		global::fatal("cannot get layer attr with null root");
	}
	auto froot = estd::must_ptr_cast<teq::iFunctor>((teq::TensptrT) root);
	auto layerattr = estd::must_cast<teq::LayerObj>(froot->get_attr(teq::layer_attr));
	return eteq::ETensor(layerattr->get_tensor(), root.get_context());
}

eteq::ETensor trail (const eteq::ETensor& root, const teq::OwnMapT& inputs)
{
	Trailer trailer(inputs);
	root->accept(trailer);
	return eteq::ETensor(estd::try_get(trailer.trailed_, root.get(), nullptr),
		root.get_context());
}

eteq::ETensor connect (const eteq::ETensor& root, const eteq::ETensor& input)
{
	return trail(root, teq::OwnMapT{
		{get_input(root).get(), (teq::TensptrT) input}});
}

eteq::ETensor deep_clone (const eteq::ETensor& root)
{
	teq::Copier kamino({get_input(root).get()});
	root->accept(kamino);
	return eteq::ETensor(kamino.clones_.at(root.get()), root.get_context());
}

eteq::VarptrsT get_storage (const eteq::ETensor& root)
{
	teq::RefMapT owner = teq::track_ownrefs(teq::TensptrsT{root});

	auto intens = get_input(root).get();
	VarExtract extra({intens});
	root->accept(extra);

	eteq::VarptrsT vars;
	vars.reserve(extra.variables_.size());
	for (auto leaf : extra.variables_)
	{
		if (auto var = std::dynamic_pointer_cast<
			eteq::Variable>(owner.at(leaf).lock()))
		{
			vars.push_back(var);
		}
	}
	return vars;
}

}

#endif
