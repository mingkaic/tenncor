#include "experimental/opt/compares.hpp"
#include "experimental/opt/rule/convert.hpp"

#include "ead/opt/rule_src.hpp"

#ifndef OPT_OPTIMIZE_HPP
#define OPT_OPTIMIZE_HPP

namespace opt
{

template <typename T>
std::vector<T> remove_duplicates (
	std::unordered_set<ade::iTensor*> priorities,
	std::vector<T> tens, const ade::ParentFinder& pfinder)
{
	if (tens.empty())
	{
		return {};
	}
	std::sort(tens.begin(), tens.end(),
		[&priorities](T& a, T& b) { return lt(priorities, a.get(), b.get()); });
	T last = tens[0];
	std::vector<T> uniques = {last};
	size_t n = tens.size();
	uniques.reserve(n - 1);
	for (size_t i = 1; i < n; ++i)
	{
		T& cur = tens[i];
		if (is_equal(last.get(), cur.get()))
		{
			logs::debugf("replacing %s", cur->to_string().c_str());
			// remove equivalent node
			auto it = pfinder.parents_.find(cur.get());
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
			tag::move_tags(last.get(), cur.get());
			tag::erase(cur.get());
		}
		else
		{
			uniques.push_back(cur);
			last = cur;
		}
	}
	return uniques;
}

void optimize (ade::TensT roots, const rule::ConversionsT& conversions)
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

	std::vector<ade::LeafptrT> immutables;
	std::vector<std::vector<ade::FuncptrT>> functors(maxheight);

	for (auto& gpair : stat.graphsize_)
	{
		auto tens = gpair.first;
		size_t height = gpair.second.upper_;
		if (0 == height)
		{
			if (tag::has_property(tens, tag::immutable_tag))
			{
				immutables.push_back(
					std::static_pointer_cast<ade::iLeaf>(
						owners.at(tens).lock()));
			}
		}
		else
		{
			functors[height - 1].push_back(
				std::static_pointer_cast<ade::iFunctor>(
					owners.at(tens).lock()));
		}
	}

	// there are no conversions for leaves
	// remove equivalent nodes
	remove_duplicates(priorities, immutables, pfinder);

	for (size_t i = 1; i < maxheight; ++i)
	{
		// remove equivalent nodes
		logs::debugf("removing duplicates for functors of height %d", i);
		std::vector<ade::FuncptrT> funcs = remove_duplicates(
			priorities, functors[i - 1], pfinder);

		// apply rule conversion to uniques

		// remove equivalent nodes in converted subgraph
	}
}

}

#endif // OPT_OPTIMIZE_HPP
