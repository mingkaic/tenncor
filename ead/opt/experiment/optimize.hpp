#include "ead/opt/experiment/compares.hpp"

#include "ead/opt/rule_src.hpp"

#ifndef OPT_OPTIMIZE_HPP
#define OPT_OPTIMIZE_HPP

namespace opt
{

namespace experiment
{

template <typename T>
void remove_duplicates (
	std::unordered_set<ade::iTensor*> priorities, std::vector<T*> tens,
	const ade::ParentFinder& pfinder, const ade::OwnerMapT& owners)
{
	if (tens.empty())
	{
		return;
	}
	std::sort(tens.begin(), tens.end(),
		[&priorities](T* a, T* b) { return lt(priorities, a, b); });
	ade::TensptrT last = nullptr;
	auto it = tens.begin();
	auto et = tens.end();
	for (; it != et && nullptr == last; ++it)
	{
		ade::TensrefT lastref = owners.at(*it);
		if (false == lastref.expired())
		{
			last = lastref.lock();
		}
	}
	for (; it != et; ++it)
	{
		auto cur = *it;
		ade::TensrefT ref = owners.at(cur);
		if (ref.expired())
		{
			continue;
		}
		if (is_equal(static_cast<T*>(last.get()), cur))
		{
			// remove equivalent node
			auto it = pfinder.parents_.find(cur);
			if (pfinder.parents_.end() != it)
			{
				for (auto& parent_pair : it->second)
				{
					auto f = static_cast<ade::iFunctor*>(parent_pair.first);
					auto& children = f->get_children();
					for (size_t i : parent_pair.second)
					{
						f->update_child({
							last,
							children[i].get_shaper(),
							children[i].map_io(),
							children[i].get_coorder()
						}, i);
					}
				}
			}
			// todo: mark parents as uninitialized, reinitialize entire graph, or uninitialize everything to begin with

			// inherit tags
			tag::move_tags(last.get(), cur);
			tag::erase(cur);
		}
		else
		{
			last = ref.lock();
		}
	}
}

template <typename T>
void optimize (ade::TensT roots, const ead::opt::ConversionsT<T>& conversions)
{
	if (roots.empty())
	{
		return;
	}

	ade::GraphStat stat;
	ade::ParentFinder pfinder;
	ade::OwnerMapT owners = ade::track_owners(roots);
	std::unordered_set<ade::iTensor*> priorities;
	for (ade::TensptrT& root : roots)
	{
		priorities.emplace(root.get());
		root->accept(stat);
		root->accept(pfinder);
	}

	// stat provides positional information:
	//		- nodes of different height will never be equivalent
	// pfinder provides adjacency information:
	//		- parents of equivalent/converted nodes will need updating
	// for each height from 0 to max:
	//		assert: every node below height is optimal and unique (non-equivalent from each other)
	//		1. delete and update equivalent nodes on the same height
	//		2. for each node at height level,
	//			apply rule conversion to non-equivalent generate converted subgraph
	//		3. delete and update equivalent nodes in converted subgraph

	std::vector<size_t> root_heights;
	root_heights.reserve(roots.size());
	std::transform(roots.begin(), roots.end(),
		std::back_inserter(root_heights),
		[&stat](ade::TensptrT& root)
		{
			return stat.graphsize_[root.get()].upper_;
		});
	// max of the maxheight of roots should be the maxheight of the whole graph
	size_t maxheight = *std::max_element(
		root_heights.begin(), root_heights.end());

	std::vector<ade::iLeaf*> leaves;
	std::vector<std::vector<ade::iFunctor*>> functors(maxheight);

	for (auto& gpair : stat.graphsize_)
	{
		auto tens = gpair.first;
		size_t height = gpair.second.upper_;
		if (0 == height)
		{
			leaves.push_back(static_cast<ade::iLeaf*>(tens));
		}
		else
		{
			functors[height - 1].push_back(static_cast<ade::iFunctor*>(tens));
		}
	}

	// there are no conversions for leaves
	// remove equivalent nodes
	std::vector<ade::iLeaf*> immutables;
	immutables.reserve(leaves.size());
	std::copy_if(leaves.begin(), leaves.end(),
		std::back_inserter(immutables), [](ade::iLeaf* leaf)
		{ return tag::has_property(leaf, tag::immutable_tag); });
	remove_duplicates(priorities, immutables, pfinder, owners);

	for (size_t i = 1; i < maxheight; ++i)
	{
		std::vector<ade::iFunctor*>& funcs = functors[i - 1];
		// remove equivalent nodes
		remove_duplicates(priorities, funcs, pfinder, owners);

		// apply rule conversion to uniques

		// remove equivalent nodes in converted subgraph
	}
}

}

}

#endif // OPT_OPTIMIZE_HPP
