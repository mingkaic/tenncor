#include "opt/optimize.hpp"

#ifdef OPT_OPTIMIZE_HPP

namespace opt
{

void replace_parents (const teq::ParentFinder& pfinder,
	teq::TensptrT target, teq::iTensor* source)
{
	teq::ParentMapT pmap;
	if (estd::get(pmap, pfinder.parents_, source))
	{
		for (auto& ppair : pmap)
		{
			auto f = static_cast<teq::iFunctor*>(ppair.first);
			for (size_t i : ppair.second)
			{
				f->update_child(target, i);
			}
		}
	}
}

teq::TensptrsT optimize (teq::TensptrsT roots,
	const CversionCtx& opts, const CustomFilters& filters)
{
	if (roots.empty())
	{
		return roots;
	}

	for (auto& filter : filters.prefilters_)
	{
		filter(roots);
	}

	teq::OwnerMapT owners = teq::track_owners(roots);
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

	std::vector<teq::FuncptrT> functors;
	functors.reserve(stat.graphsize_.size());
	for (auto& gpair : stat.graphsize_)
	{
		if (gpair.second.upper_ > 0)
		{
			functors.push_back(std::static_pointer_cast<teq::iFunctor>(
				owners.at(gpair.first).lock()));
		}
	}
	std::sort(functors.begin(), functors.end(),
		[&stat](teq::FuncptrT a, teq::FuncptrT b)
		{
			return stat.graphsize_[a.get()].upper_ <
				stat.graphsize_[b.get()].upper_;
		});

	ParentReplF parent_replacer =
	[&](teq::TensptrT dest, teq::iTensor* src)
	{
		replace_parents(pfinder, dest, src);
		std::vector<size_t> ridx;
		if (estd::get(ridx, rindices, src))
		{
			for (size_t ri : ridx)
			{
				roots[ri] = dest;
			}
		}
	};

	MatchCtxT runtime_ctx;
	CversionsT conversions;
	// there are no conversions for leaves
	for (teq::FuncptrT func : functors)
	{
		// todo: streamline this pipeline
		bool stop_filtering = false;
		// apply prefiltering
		for (size_t i = 0, n = filters.prenode_filters_.size();
			i < n && false == stop_filtering; ++i)
		{
			auto converted = filters.prenode_filters_[i](func, parent_replacer);
			if (converted != func)
			{
				if (auto fconv = std::dynamic_pointer_cast<teq::iFunctor>(converted))
				{
					func = fconv;
				}
				else
				{
					stop_filtering = true;
				}
			}
		}
		if (stop_filtering)
		{
			continue;
		}

		// apply optimization
		if (auto converted = opts.optimize(runtime_ctx, func.get()))
		{
			// todo: debug log converted

			// don't touch functors until after all nodes are visited
			parent_replacer(converted, func.get());

			if (auto fconv = std::dynamic_pointer_cast<teq::iFunctor>(converted))
			{
				func = fconv;
			}
			else
			{
				stop_filtering = true;
			}
		}
		if (stop_filtering)
		{
			continue;
		}

		// apply postfiltering
		for (size_t i = 0, n = filters.postnode_filters_.size();
			i < n && false == stop_filtering; ++i)
		{
			auto converted = filters.postnode_filters_[i](func, parent_replacer);
			if (converted != func)
			{
				if (auto fconv = std::dynamic_pointer_cast<teq::iFunctor>(converted))
				{
					func = fconv;
				}
				else
				{
					stop_filtering = true;
				}
			}
		}
	}

	for (auto& filter : filters.postfilters_)
	{
		filter(roots);
	}
	return roots;
}

}

#endif // OPT_OPTIMIZE_HPP
