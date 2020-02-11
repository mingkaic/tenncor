
#ifndef EXPERIMENTAL_OPT_GRAPH_HPP
#define EXPERIMENTAL_OPT_GRAPH_HPP

#include <list>

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
		owners_(convert_ownermap(teq::track_owners(roots)))
	{
		teq::ParentFinder pfinder;
		for (auto root : roots)
		{
			root->accept(pfinder);
			root->accept(pbuilder_);
		}
		parents_ = pfinder.parents_;
		for (auto& parent : parents_)
		{
			if (parent.second.empty())
			{
				roots_.push_back(parent.first);
			}
		}

		query::search::populate_itable(sindex_, pbuilder_.paths_);
	}

	teq::TensptrsT find (const query::Node& condition) const
	{
		query::QResultsT results;
		query::Query(sindex_).where(
			std::make_shared<query::Node>(condition)).exec(results);
		teq::TensptrsT outs;
		outs.reserve(results.size());
		std::transform(results.begin(), results.end(),
			std::back_inserter(outs),
			[this](const query::QueryResult& result)
			{
				return owners_.at(result.root_);
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
				auto children = f->get_children();
				for (auto child : children)
				{
					clean_children[child.get()].emplace(src);
				}
			}
			if (estd::has(parents_, src))
			{
				// for ancestors of src, clean pbuilder_.paths_ info
				std::list<teq::iTensor*> q = {src};
				teq::TensSetT clean_pars;
				while (false == q.empty())
				{
					teq::iTensor* parent = q.front();
					q.pop_front();
					pbuilder_.paths_.erase(parent);
					if (estd::has(parents_, parent))
					{
						teq::ParentMapT& pmap = parents_.at(parent);
						for (auto& ppair : pmap)
						{
							if (false == estd::has(
								clean_pars, ppair.first))
							{
								q.push_back(ppair.first);
								clean_pars.emplace(ppair.first);
							}
						}
					}
				}

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

			// update owners and track changes regarding target
			owners_.emplace(target.get(), target);
			changes_[src] = target;
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
		// update pbuilder_
		for (auto root : roots_)
		{
			if (estd::has(changes_, root))
			{
				root = changes_.at(root).get();
			}
			root->accept(pbuilder_);
		}
		// update sindex_
		sindex_.clear();
		query::search::populate_itable(sindex_, pbuilder_.paths_);
	}

	teq::TensT roots_;

	OwnersT owners_; // todo: cleanup everything properly instead of keeping dangling leaves

	// sindex_ built from pbuilder_.paths_
	query::search::OpTrieT sindex_;

	teq::TensMapT<teq::ParentMapT> parents_;

	teq::TensMapT<teq::TensptrT> changes_;

private:
	query::search::OpPathBuilder pbuilder_;
};

}

#endif // EXPERIMENTAL_OPT_GRAPH_HPP
