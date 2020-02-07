
#ifndef EXPERIMENTAL_OPT_GRAPH_HPP
#define EXPERIMENTAL_OPT_GRAPH_HPP

#include <queue>

#include "query/query.hpp"

namespace opt
{

struct GraphInfo final
{
	GraphInfo (teq::TensptrsT roots) :
		owners_(teq::track_owners(roots))
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
				return owners_.at(result.root_).lock();
			});
		return outs;
	}

	void replace (const teq::TensMapT<teq::TensptrT>& converts)
	{
		teq::ParentMapT pmap;
		for (const auto& convert : converts)
		{
			teq::iTensor* src = convert.first;
			teq::TensptrT target = convert.second;
			if (estd::get(pmap, parents_, src))
			{
				// for ancestors of src, clean pbuilder_.paths_ info
				std::queue<teq::iTensor*> q;
				q.push(src);
				while (false == q.empty())
				{
					teq::iTensor* parent = q.front();
					pbuilder_.paths_.erase(parent);
					if (estd::get(pmap, parents_, parent))
					{
						for (auto& ppair : pmap)
						{
							q.push(ppair.first);
						}
					}
					q.pop();
				}

				// update parents_
				teq::ParentMapT& tmap = parents_[target.get()];
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
		}
		// cleanup parents_
		for (const auto& convert : converts)
		{
			// stop src children from referencing src
			teq::iTensor* src = convert.first;
			if (auto f = dynamic_cast<teq::iFunctor*>(src))
			{
				auto children = f->get_children();
				for (teq::TensptrT child : children)
				{
					parents_[child.get()].erase(src);
				}
			}
			// remove src references in parents_
			parents_.erase(src);
		}
		// update pbuilder_
		for (auto root : roots_)
		{
			root->accept(pbuilder_);
		}
		// update sindex_
		sindex_.clear();
		query::search::populate_itable(sindex_, pbuilder_.paths_);
	}

	teq::TensT roots_;

	teq::OwnerMapT owners_;

	// sindex_ built from pbuilder_.paths_
	query::search::OpTrieT sindex_;

	teq::TensMapT<teq::ParentMapT> parents_;

private:
	query::search::OpPathBuilder pbuilder_;
};

}

#endif // EXPERIMENTAL_OPT_GRAPH_HPP
