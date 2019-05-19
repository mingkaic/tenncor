#include <list>
#include <unordered_set>

#include "ade/traveler.hpp"

#include "ead/constant.hpp"
#include "ead/functor.hpp"

#ifndef EAD_SESSION_HPP
#define EAD_SESSION_HPP

namespace ead
{

using TensSetT = std::unordered_set<ade::iTensor*>;

struct iSession
{
	virtual ~iSession (void) = default;

	virtual void track (ade::iTensor* root) = 0;

	/// update all nodes related to updated, if updated set is empty
	/// update all nodes related to the leaves (so everyone)
	/// ignore all nodes dependent on ignores including the ignored nodes
	virtual void update (TensSetT updated = {}, TensSetT ignores = {}) = 0;
};

// for each leaf node, iteratively update the parents
// don't update parent node if it is part of ignored set
struct Session final : public iSession
{
	void track (ade::iTensor* root) override
	{
		ade::ParentFinder pfinder;
		root->accept(pfinder);
		root->accept(stat_);
		auto& statmap = stat_.graphsize_;

		std::list<ade::iOperableFunc*> all_ops;
		for (auto& statpair : statmap)
		{
			if (0 < statpair.second.upper_)
			{
				// ensure we only track operable functors
				auto op = dynamic_cast<ade::iOperableFunc*>(statpair.first);
				if (nullptr == op)
				{
					logs::fatalf("cannot track non-operable functor %s",
						statpair.first->to_string().c_str());
				}
				all_ops.push_back(op);
			}
		}
		all_ops.sort(
			[&statmap](ade::iOperableFunc* a, ade::iOperableFunc* b)
			{
				return statmap[a].upper_ < statmap[b].upper_;
			});
		requirements_.clear();
		for (ade::iOperableFunc* op : all_ops)
		{
			auto& args = op->get_children();
			std::unordered_set<ade::iTensor*> unique_children;
			for (const ade::FuncArg& arg : args)
			{
				auto tens = arg.get_tensor().get();
				if (0 < statmap[tens].upper_) // ignore leaves
				{
					unique_children.emplace(tens);
				}
			}
			requirements_.push_back({op, unique_children.size()});
		}

		for (auto& assocs : pfinder.parents_)
		{
			for (ade::iTensor* parent : assocs.second)
			{
				parents_[assocs.first].emplace(static_cast<ade::iOperableFunc*>(parent));
			}
		}
	}

	struct SizeTDefaultZero
	{
		size_t d = 0;
	};

	void update (TensSetT updated = {}, TensSetT ignores = {}) override
	{
		std::unordered_map<ade::iOperableFunc*,SizeTDefaultZero> fulfilments;
		for (ade::iTensor* unodes : updated)
		{
			auto& node_parents = parents_[unodes];
			for (auto& node_parent : node_parents)
			{
				++fulfilments[node_parent].d;
			}
		}
		// assert: ignored nodes and its dependers will never fulfill requirement
		for (auto& op : requirements_)
		{
			// fulfilled and not ignored
			if (fulfilments[op.first].d >= op.second &&
				ignores.end() == ignores.find(op.first))
			{
				op.first->update();
				auto& op_parents = parents_[op.first];
				for (auto& op_parent : op_parents)
				{
					++fulfilments[op_parent].d;
				}
			}
		}
	}

	ade::GraphStat stat_;

	std::unordered_map<ade::iTensor*,
		std::unordered_set<ade::iOperableFunc*>> parents_;

	std::vector<std::pair<ade::iOperableFunc*,size_t>> requirements_; // todo: test minimal requirements
};

struct InteractiveSession final : public iSession
{
	void track (ade::iTensor* root) override
	{
		sess_.track(root);

		// for (auto& statpair : stat_.graphsize_)
		// {
		// 	auto tens = statpair.first;
		// 	if (node_ids_.emplace(tens, node_ids_.size().second))
		// 	{
		// 		// add to POST request
		// 	}
		// }

		// for (auto ppair : parents_)
		// {
		// 	for (ade::iTensor* parent : ppair)
		// 	{
		// 		Edge edge{
		// 			node_ids_[parent],
		// 			node_ids_[ppair.first],
		// 			"parent-child",
		// 		};
		// 	}
		// }
	}

	void update (TensSetT updated = {}, TensSetT ignores = {}) override
	{
		//
	}

	Session sess_;

	std::unordered_map<ade::iTensor*,size_t> node_ids_;
};

}

#endif // EAD_SESSION_HPP
