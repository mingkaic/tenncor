#include "opt/rmdups.hpp"
#include "opt/optimize.hpp"

#ifdef OPT_OPTIMIZE_HPP

namespace opt
{

ade::TensT optimize (ade::TensT roots, const OptCtx& opts)
{
	if (roots.empty())
	{
		return roots;
	}

	// stat provides positional information:
	//		- nodes of different height will never be equivalent
	// pfinder provides adjacency information:
	//		- parents of equivalent/converted nodes will need updating
	// adjgroups provides group information

	// 1. remove duplicates in the graph to avoid duplicate conversion checks
	// 2. perform conversions
	// 3. remove duplicates in the graph in case of duplicates in conversions

	{
		HFunctorsT functors;
		// step 1:
		{
			ImmutablesT immutables;
			populate_graph(immutables, functors, roots);
			remove_all_duplicates(roots, immutables, functors);
		}

		// step 2:
		ade::GraphStat stat;
		ade::ParentFinder pfinder;
		Matcher matcher(opts.voters_);
		tag::AdjacentGroups adjgroups;
		std::unordered_map<ade::iTensor*,std::vector<size_t>> rindices;
		for (size_t i = 0, n = roots.size(); i < n; ++i)
		{
			ade::TensptrT& root = roots[i];
			root->accept(stat);
			root->accept(pfinder);
			root->accept(matcher);
			root->accept(adjgroups);
			rindices[root.get()].push_back(i);
		}

		{
			tag::SubgraphsT subgraphs;
			tag::beautify_groups(subgraphs, adjgroups);
			// filter subgraphs for "heads" to add to matcher.group_head_
			std::unordered_map<tag::SgraphptrT,ade::iTensor*> revhead;
			for (auto& sgpair : subgraphs)
			{
				if (util::has(revhead, sgpair.second))
				{
					ade::iTensor*& oldhead = revhead[sgpair.second];
					if (stat.graphsize_[sgpair.first].upper_ >
						stat.graphsize_[oldhead].upper_)
					{
						// add sgpair.first as head if it has greater maxheight
						revhead[sgpair.second] = sgpair.first;
					}
				}
				else
				{
					revhead.emplace(sgpair.second, sgpair.first);
				}
			}
			for (auto& revpair : revhead)
			{
				matcher.group_head_[revpair.second] = revpair.first;
			}
		}

		// there are no conversions for leaves
		const ConvptrT& const_conv = opts.const_conv_;
		for (auto& funcs : functors)
		{
			for (ade::FuncptrT func : funcs)
			{
				ade::TensptrT converted = nullptr;
				ade::Shape shape = func->shape();
				auto& cands = matcher.candidates_[func.get()];
				// select the best candidate (smallest conversion)
				// currently first come first serve (todo: implement)
				for (auto& candpair : cands)
				{
					if (CAND_TYPE::CONVRT == candpair.first.type_)
					{
						CtxsT& ctxs = candpair.second;
						ContexT ctx;
						if (ctxs.size() > 1)
						{
							logs::warn("ambiguous context");
						}
						if (ctxs.size() > 0)
						{
							ctx = *(ctxs.begin());
						}
						const ConvptrT& conv = opts.converts_.at(
							candpair.first.reference_);
						logs::debugf("converting to %s",
							conv->to_string().c_str());
						converted = conv->build(ctx, shape);
						break;
					}
					else if (CAND_TYPE::CONST == candpair.first.type_)
					{
						ContexT ctx = {
							{func->to_string(), {func}},
						};
						converted = const_conv->build(ctx, shape);
						break;
					}
				}

				if (nullptr != converted)
				{
					replace_parents(pfinder, func.get(), converted);
					auto it = rindices.find(func.get());
					if (rindices.end() != it)
					{
						for (size_t ri : it->second)
						{
							roots[ri] = converted;
						}
					}
				}
			}
		}
	}

	// step 3:
	HFunctorsT functors;
	ImmutablesT immutables;
	populate_graph(immutables, functors, roots);
	remove_all_duplicates(roots, immutables, functors);

	return roots;
}

}

#endif // OPT_OPTIMIZE_HPP
