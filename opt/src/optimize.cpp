#include "opt/rmdups.hpp"
#include "opt/optimize.hpp"

#ifdef OPT_OPTIMIZE_HPP

namespace opt
{

void optimize (ade::TensT roots, const OptCtx& opts)
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
	// 2. perform conversions
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
			remove_all_duplicates(immutables, functors, rset);
		}

		// step 2:
		ade::ParentFinder pfinder;
		Matcher matcher(opts.voters_);
		for (ade::iTensor* root : rset)
		{
			root->accept(pfinder);
			root->accept(matcher);
		}

		// there are no conversions for leaves
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
						converted = conv->build(ctx, shape);
						break;
					}
				}

				if (nullptr != converted)
				{
					replace_parents(pfinder, func.get(), converted);
				}
			}
		}
	}

	// step 3:
	HFunctorsT functors;
	ImmutablesT immutables;
	populate_graph(immutables, functors, roots);
	remove_all_duplicates(immutables, functors, rset);
}

}

#endif // OPT_OPTIMIZE_HPP
