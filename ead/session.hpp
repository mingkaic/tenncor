#include <list>
#include <unordered_set>

#include "ade/traveler.hpp"

#include "opt/optimize.hpp"

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

	virtual void track (ade::TensT roots) = 0;

	/// update all nodes related to updated, if updated set is empty
	/// update all nodes related to the leaves (so everyone)
	/// ignore all nodes dependent on ignores including the ignored nodes
	virtual void update (TensSetT updated = {}, TensSetT ignores = {}) = 0;

	virtual void update_target (TensSetT target, TensSetT updated = {}) = 0;
};

struct SizeT
{
	size_t d = 0;

	operator size_t() const { return d; }
};

// todo: give this more reasons for existence
struct Traveler final : public ade::iTraveler
{
	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		visited_.emplace(leaf);
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (false == estd::has(visited_, func))
		{
			visited_.emplace(func);
			auto& children = func->get_children();
			for (auto& child : children)
			{
				child.get_tensor()->accept(*this);
			}
		}
	}

	TensSetT visited_;
};

// for each leaf node, iteratively update the parents
// don't update parent node if it is part of ignored set
struct Session final : public iSession
{
	void track (ade::TensT roots) override
	{
		tracked_.insert(roots.begin(), roots.end());
		ade::ParentFinder pfinder;
		for (ade::TensptrT& root : roots)
		{
			root->accept(pfinder);
			root->accept(stat_);
		}
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
			TensSetT unique_children;
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
			for (auto& parent_pair : assocs.second)
			{
				parents_[assocs.first].emplace(
					static_cast<ade::iOperableFunc*>(parent_pair.first));
			}
		}
	}

	void update (TensSetT updated = {}, TensSetT ignores = {}) override
	{
		std::unordered_map<ade::iOperableFunc*,SizeT> fulfilments;
		for (ade::iTensor* unodes : updated)
		{
			auto& node_parents = parents_[unodes];
			for (auto& node_parent : node_parents)
			{
				++fulfilments[node_parent].d;
			}
		}
		// ignored nodes and its dependers will never fulfill requirement
		for (auto& op : requirements_)
		{
			// fulfilled and not ignored
			if (fulfilments[op.first].d >= op.second &&
				false == estd::has(ignores, op.first))
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

	void update_target (TensSetT target, TensSetT updated = {}) override
	{
		Traveler targetted;
		for (auto& tens : target)
		{
			tens->accept(targetted);
		}
		std::unordered_map<ade::iOperableFunc*,SizeT> fulfilments;
		for (ade::iTensor* unodes : updated)
		{
			auto& node_parents = parents_[unodes];
			for (auto& node_parent : node_parents)
			{
				++fulfilments[node_parent].d;
			}
		}
		// ignored nodes and its dependers will never fulfill requirement
		for (auto& op : requirements_)
		{
			// is relevant to target, is fulfilled and not ignored
			if (estd::has(targetted.visited_, op.first) &&
				fulfilments[op.first].d >= op.second)
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

	void optimize (const opt::OptCtx& rules)
	{
		ade::TensT tracked(tracked_.begin(), tracked_.end());
		opt::optimize(tracked, rules);
		stat_.graphsize_.clear();
		parents_.clear();
		track(tracked);
	}

	std::unordered_set<ade::TensptrT> tracked_;

	ade::GraphStat stat_;

	std::unordered_map<ade::iTensor*,
		std::unordered_set<ade::iOperableFunc*>> parents_;

	std::vector<std::pair<ade::iOperableFunc*,size_t>> requirements_; // todo: test minimal requirements
};

}

#endif // EAD_SESSION_HPP
