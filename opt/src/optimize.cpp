#include "opt/rmdups.hpp"
#include "opt/optimize.hpp"

#ifdef OPT_OPTIMIZE_HPP

namespace opt
{

teq::TensT optimize (teq::TensT roots, const OptCtx& opts)
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
		CstConvertF const_conv = opts.const_conv_;
		Matcher matcher(opts.voters_);
		matcher.scalarize_ =
			[&const_conv](teq::iTensor* tens) -> std::string
			{
				std::string out;
				if (auto cst = const_conv(tens))
				{
					out = cst->to_string();
				}
				else
				{
					out = tens->to_string();
				}
				return out;
			};
		teq::GraphStat stat;
		teq::ParentFinder pfinder;
		std::unordered_map<teq::iTensor*,std::vector<size_t>> rindices;
		for (size_t i = 0, n = roots.size(); i < n; ++i)
		{
			teq::TensptrT& root = roots[i];
			root->accept(stat);
			root->accept(pfinder);
			rindices[root.get()].push_back(i);
		}

		{
			tag::AdjMapT adjs;
			tag::adjacencies(adjs, roots);

			tag::SubgraphAssocsT subgraphs;
			tag::beautify_groups(subgraphs, adjs);
			tag::filter_head(matcher.group_head_, subgraphs);
		}

		// there are no conversions for leaves
		for (auto& funcs : functors)
		{
			for (teq::FuncptrT func : funcs)
			{
				// although matcher recursively applies to functor children,
				// it's easier to evaluate near conversion to avoid tracking state changes
				func->accept(matcher);

				teq::TensptrT converted = nullptr;
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
						converted = conv->build(ctx, func->shape());
						break;
					}
					else if (CAND_TYPE::CONST == candpair.first.type_)
					{
						converted = const_conv(func.get());
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
