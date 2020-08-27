
#include "tenncor/opt/duplicates.hpp"

#ifdef TENNCOR_OPT_DUPLICATES_HPP

namespace eteq
{

void merge_dups (opt::GraphInfo& graph, EqualF equals)
{
	teq::GraphStat stat;
	for (teq::TensptrT root : graph.roots_)
	{
		root->accept(stat);
	}

	std::vector<teq::LeafptrT> csts;
	std::vector<teq::FuncptrT> functors;
	for (auto& gpair : stat.graphsize_)
	{
		auto tens = gpair.first;
		size_t height = gpair.second.upper_;
		if (0 == height)
		{
			auto leaf = std::static_pointer_cast<teq::iLeaf>(
				graph.get_owner(tens));
			if (teq::IMMUTABLE == leaf->get_usage())
			{
				csts.push_back(leaf);
			}
		}
		else
		{
			functors.push_back(std::static_pointer_cast<teq::iFunctor>(
				graph.get_owner(tens)));
		}
	}

	teq::OwnMapT converts;
	// remove equivalent nodes
	if (csts.size() > 0)
	{
		global::debug("removing immutable duplicates");
		std::sort(csts.begin(), csts.end(),
			[](teq::LeafptrT a, teq::LeafptrT b)
			{
				return a->to_string() < b->to_string();
			});
		teq::LeafptrT cmp = csts[0];
		for (size_t i = 1, n = csts.size(); i < n; ++i)
		{
			if (equals(cmp, csts[i]))
			{
				// mark equivalent node
				converts.emplace(csts[i].get(), cmp);
			}
			else
			{
				cmp = csts[i];
			}
		}
	}

	if (functors.size() > 0)
	{
		global::debug("removing functor duplicates");
		std::sort(functors.begin(), functors.end(),
			[&stat](teq::FuncptrT a, teq::FuncptrT b)
			{
				size_t lheight = stat.graphsize_[a.get()].upper_;
				size_t rheight = stat.graphsize_[b.get()].upper_;
				if (lheight == rheight)
				{
					return a->to_string() < b->to_string();
				}
				return lheight < rheight;
			});
		teq::FuncptrT cmp = functors[0];
		for (size_t i = 1, n = functors.size(); i < n; ++i)
		{
			if (equals(cmp, functors[i]))
			{
				// mark equivalent node
				converts.emplace(functors[i].get(), cmp);
			}
			else
			{
				cmp = functors[i];
			}
		}
	}
	graph.replace(converts);
}

void merge_dups (opt::GraphInfo& graph)
{
	Hasher hasher;
	for (auto& root : graph.roots_)
	{
		root->accept(hasher);
	}
	merge_dups(graph,
		[&](teq::TensptrT a, teq::TensptrT b)
		{
			return hasher.at(a.get()) == hasher.at(b.get());
		});
}

}

#endif
