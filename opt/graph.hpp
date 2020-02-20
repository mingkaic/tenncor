
#ifndef OPT_GRAPH_HPP
#define OPT_GRAPH_HPP

#include <list>

#include "query/query.hpp"

namespace opt
{

// Map of tensors to shared pointers
// to actually own tensors to avoid early deletion
using OwnersT = teq::TensMapT<teq::TensptrT>;

OwnersT convert_ownermap (const teq::OwnerMapT& omap);

struct UnindexedGraph final
{
	UnindexedGraph (teq::TensptrsT roots) :
		owners_(convert_ownermap(teq::track_owners(roots)))
	{
		teq::ParentFinder pfinder;
		for (auto root : roots)
		{
			root->accept(pfinder);
		}
		parents_ = pfinder.parents_;
		for (auto& parent : parents_)
		{
			if (parent.second.empty())
			{
				roots_.push_back(parent.first);
			}
		}
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
		for (teq::iTensor*& root : roots_)
		{
			if (estd::has(converts, root))
			{
				root = converts.at(root).get();
			}
		}
	}

	teq::TensT roots_;

	OwnersT owners_; // todo: cleanup everything properly instead of keeping dangling leaves

	teq::TensMapT<teq::ParentMapT> parents_;
};

struct GraphInfo final
{
	GraphInfo (const UnindexedGraph& base) : base_(base)
	{
		for (auto root : base_.roots_)
		{
			root->accept(pbuilder_);
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
				return base_.owners_.at(result.root_);
			});
		return outs;
	}

	void replace (const teq::TensMapT<teq::TensptrT>& converts)
	{
		teq::TensMapT<teq::TensSetT> clean_children;
		for (const auto& convert : converts)
		{
			teq::iTensor* src = convert.first;
			if (estd::has(base_.parents_, src))
			{
				// for ancestors of src, clean pbuilder_.paths_ info
				std::list<teq::iTensor*> q = {src};
				teq::TensSetT clean_pars;
				while (false == q.empty())
				{
					teq::iTensor* parent = q.front();
					q.pop_front();
					pbuilder_.paths_.erase(parent);
					if (estd::has(base_.parents_, parent))
					{
						teq::ParentMapT& pmap = base_.parents_.at(parent);
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
			}
		}
		base_.replace(converts);
		// update pbuilder_
		for (auto root : base_.roots_)
		{
			root->accept(pbuilder_);
		}
		// update sindex_
		sindex_.clear();
		query::search::populate_itable(sindex_, pbuilder_.paths_);
	}

	teq::TensT get_roots (void) const
	{
		return base_.roots_;
	}

	const OwnersT& get_owners (void) const
	{
		return base_.owners_;
	}

	teq::TensptrT get_owner (teq::iTensor* ptr) const
	{
		return estd::try_get(base_.owners_, ptr, nullptr);
	}

	// sindex_ built from pbuilder_.paths_
	query::search::OpTrieT sindex_;

private:
	UnindexedGraph base_;

	query::search::OpPathBuilder pbuilder_;
};

}

#endif // OPT_GRAPH_HPP
