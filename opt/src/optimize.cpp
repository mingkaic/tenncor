#include "opt/optimize.hpp"

#ifdef OPT_OPTIMIZE_HPP

namespace opt
{

void replace_parents (const teq::ParentFinder& pfinder,
	teq::TensptrT target, teq::iTensor* source,
	tag::TagRegistry& registry)
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
	registry.move_tags(target, source);
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

	MatchCtxT runtime_ctx;
	CversionsT conversions;
	// there are no conversions for leaves
	for (teq::FuncptrT func : functors)
	{
		if (auto converted = opts.optimize(runtime_ctx, func.get()))
		{
			// todo: debug log converted

			// don't touch functors until after all nodes are visited
			replace_parents(pfinder, converted, func.get());

			std::vector<size_t> ridx;
			if (estd::get(ridx, rindices, func.get()))
			{
				for (size_t ri : ridx)
				{
					roots[ri] = converted;
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

void optimize (teq::iSession& sess,
	const CversionCtx& opts, const CustomFilters& filters)
{
	teq::TensptrSetT tracked_set = sess.get_tracked();
	teq::TensptrsT tracked(tracked_set.begin(), tracked_set.end());
	optimize(tracked, opts, filters);
	sess.clear();
	sess.track(tracked);
}

}

#endif // OPT_OPTIMIZE_HPP
