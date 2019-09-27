#include "opt/rmdups.hpp"

#ifdef OPT_RMDUPS_HPP

namespace opt
{

void replace_parents (const teq::ParentFinder& pfinder,
	teq::iTensor* source, teq::TensptrT target)
{
	auto it = pfinder.parents_.find(source);
	if (pfinder.parents_.end() != it)
	{
		for (auto& parent_pair : it->second)
		{
			auto f = static_cast<teq::iFunctor*>(parent_pair.first);
			auto& children = f->get_children();
			for (size_t i : parent_pair.second)
			{
				f->update_child({
					target,
					children[i].get_shaper(),
					children[i].map_io(),
					children[i].get_coorder()
				}, i);
			}
		}
	}
}

void populate_graph (ImmutablesT& immutables, HFunctorsT& functors,
	const teq::TensT& roots)
{
	teq::OwnerMapT owners = teq::track_owners(roots);
	teq::GraphStat stat;
	for (teq::TensptrT root : roots)
	{
		root->accept(stat);
	}

	std::vector<size_t> root_heights;
	root_heights.reserve(roots.size());
	std::transform(roots.begin(), roots.end(),
		std::back_inserter(root_heights),
		[&stat](const teq::TensptrT& root)
		{
			return stat.graphsize_[root.get()].upper_;
		});
	// max of the maxheight of roots should be the maxheight of the whole graph
	size_t maxheight = *std::max_element(
		root_heights.begin(), root_heights.end());
	functors = HFunctorsT(maxheight);
	for (auto& gpair : stat.graphsize_)
	{
		auto tens = gpair.first;
		size_t height = gpair.second.upper_;
		if (0 == height)
		{
			if (tag::get_property_reg().has_property(tens, tag::immutable_tag))
			{
				immutables.push_back(
					std::static_pointer_cast<teq::iLeaf>(
						owners.at(tens).lock()));
			}
		}
		else
		{
			functors[height - 1].push_back(
				std::static_pointer_cast<teq::iFunctor>(
					owners.at(tens).lock()));
		}
	}
}

void remove_all_duplicates (teq::TensT& roots,
	ImmutablesT& immutables, HFunctorsT& functors)
{
	// remove equivalent nodes
	teq::ParentFinder pfinder;
	for (teq::TensptrT& root : roots)
	{
		root->accept(pfinder);
	}
	logs::debug("removing immutable duplicates");
	immutables = remove_duplicates(roots, immutables, pfinder);
	for (size_t i = 0, n = functors.size(); i < n; ++i)
	{
		// assert that every node below height is unique
		logs::debugf("removing functor duplicates at height %d", i + 1);
		functors[i] = remove_duplicates(roots, functors[i], pfinder);
	}
}

}

#endif
