
#ifndef OPT_GRAPH_HPP
#define OPT_GRAPH_HPP

#include "query/query.hpp"

namespace opt
{

// Map of tensors to shared pointers
// to actually own tensors to avoid early deletion
using OwnersT = teq::TensMapT<teq::TensptrT>;

OwnersT convert_ownermap (const teq::OwnerMapT& omap);

struct GraphInfo final
{
	GraphInfo (teq::TensptrsT roots) :
		roots_(roots),
		owners_(convert_ownermap(teq::track_owners(roots)))
	{
		teq::ParentFinder pfinder;
		teq::multi_visit(pfinder, roots_);
		teq::multi_visit(sindex_, roots_);
		parents_ = pfinder.parents_;
	}

	teq::TensptrsT find (const query::Node& condition) const
	{
		query::QResultsT results = sindex_.match(condition);
		teq::TensptrsT outs;
		outs.reserve(results.size());
		std::transform(results.begin(), results.end(),
			std::back_inserter(outs),
			[this](const query::QueryResult& result)
			{
				return estd::must_getf(owners_, result.root_,
					"can't find reference to %s",
					result.root_->to_string().c_str());
			});
		return outs;
	}

	void replace (const teq::TensMapT<teq::TensptrT>& converts)
	{
		teq::TensMapT<teq::TensSetT> clean_children;
		for (const auto& convert : converts)
		{
			teq::iTensor* src = convert.first;
			teq::TensptrT target = convert.second;
			if (auto f = dynamic_cast<teq::iFunctor*>(src))
			{
				auto children = f->get_dependencies();
				for (auto child : children)
				{
					clean_children[child.get()].emplace(src);
				}
			}
			if (estd::has(parents_, src))
			{
				// update parents_
				teq::ParentMapT& tmap = parents_[target.get()];
				teq::ParentMapT& pmap = parents_.at(src);
				for (auto& ppair : pmap)
				{
					auto f = static_cast<teq::iFunctor*>(ppair.first);
					for (size_t i : ppair.second)
					{
						f->update_child(target, i);
					}
					tmap.emplace(ppair);
				}
			}
			parents_.erase(src);
			// update owners and track changes regarding target
			owners_.emplace(target.get(), target);
			sindex_.erase(src);
			target->accept(sindex_);
		}
		// cleanup parents_
		for (auto childpair : clean_children)
		{
			for (teq::iTensor* src : childpair.second)
			{
				// remove src children references in parents_
				parents_[childpair.first].erase(src);
				// remove src references in parents_
				parents_.erase(src);
			}
		}
		for (teq::TensptrT& root : roots_)
		{
			if (estd::has(converts, root.get()))
			{
				root = converts.at(root.get());
			}
		}
	}

	teq::TensptrsT get_roots (void) const
	{
		return roots_;
	}

	const OwnersT& get_owners (void) const
	{
		return owners_;
	}

	teq::TensptrT get_owner (teq::iTensor* ptr) const
	{
		return estd::try_get(owners_, ptr, nullptr);
	}

	// sindex_ built from pbuilder_.paths_
	query::Query sindex_;

	teq::TensptrsT roots_;

	OwnersT owners_; // todo: cleanup everything properly instead of keeping dangling leaves

	teq::TensMapT<teq::ParentMapT> parents_;
};

}

#endif // OPT_GRAPH_HPP
