#include "experimental/opt/optimize.hpp"

#ifdef OPT_OPTIMIZE_HPP

namespace opt
{

void replace_parents (const ade::ParentFinder& pfinder,
	ade::iTensor* source, ade::TensptrT target)
{
	auto it = pfinder.parents_.find(source);
	if (pfinder.parents_.end() != it)
	{
		for (auto& parent_pair : it->second)
		{
			auto f = static_cast<ade::iFunctor*>(parent_pair.first);
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

using ImmutablesT = std::vector<ade::LeafptrT>;

using HFunctorsT = std::vector<std::vector<ade::FuncptrT>>;

// identify immutable leaves and organize functors by maxheight
static void populate_graph (ImmutablesT& immutables, HFunctorsT& functors,
	const ade::TensT& roots)
{
	ade::OwnerMapT owners = ade::track_owners(roots);
	ade::GraphStat stat;
	for (ade::TensptrT root : roots)
	{
		root->accept(stat);
	}

	std::vector<size_t> root_heights;
	root_heights.reserve(roots.size());
	std::transform(roots.begin(), roots.end(),
		std::back_inserter(root_heights),
		[&stat](const ade::TensptrT& root)
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
}

// delete and update equivalent immutable leaves and functors
static void remove_duplicates (ImmutablesT& immutables, HFunctorsT& functors,
	const std::unordered_set<ade::iTensor*>& roots)
{
	// remove equivalent nodes
	ade::ParentFinder pfinder;
	for (ade::iTensor* root : roots)
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

void optimize (ade::TensT roots, const rule::ConversionsT& conversions)
{
	if (roots.empty())
	{
		return;
	}

	// stat provides positional information:
	//		- nodes of different height will never be equivalent
	// pfinder provides adjacency information:
	//		- parents of equivalent/converted nodes will need updating
	// adjgroups provides group information

	// 1. remove duplicates in the graph to avoid duplicate conversion checks
	// 2. perform node-based conversions then group-based conversions
	// 3. remove duplicates in the graph in case of duplicates in conversions

	// preprocessing
	std::unordered_set<ade::iTensor*> rset;
	for (ade::TensptrT& root : roots)
	{
		rset.emplace(root.get());
	}

	{
		HFunctorsT functors;
		// step 1:
		{
			ImmutablesT immutables;
			populate_graph(immutables, functors, roots);
			remove_duplicates(immutables, functors, rset);
		}

		// step 2:
		// there are no conversions for leaves
		ade::ParentFinder pfinder;
		for (ade::iTensor* root : rset)
		{
			root->accept(pfinder);
		}

		tag::AdjacentGroups adjgroups;
		for (ade::TensptrT& root : roots)
		{
			root->accept(adjgroups);
		}
		tag::SubgraphsT subs;
		tag::beautify_groups(subs, adjgroups);

		// only need to look at nodes at or above minimum height
		for (auto& funcs : functors)
		{
			for (ade::FuncptrT func : funcs)
			{
				for (const rule::Conversion& conversion : conversions)
				{
					// todo: consider reducing functors with only constant arguments
					if (auto conv = conversion.convert(subs, func))
					{
						// converted
						std::string conversion_label =
							conversion.writer_->to_string() + "=>" +
							conversion.builder_->to_string();
						logs::infof("applying %s", conversion_label.c_str());
						replace_parents(pfinder, func.get(), conv);
					}
				}
			}
		}
	}

	// step 3:
	HFunctorsT functors;
	ImmutablesT immutables;
	populate_graph(immutables, functors, roots);
	remove_duplicates(immutables, functors, rset);
}

}

#endif // OPT_OPTIMIZE_HPP
