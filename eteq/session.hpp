#include <list>
#include <unordered_set>

#include "teq/traveler.hpp"

#include "opt/optimize.hpp"

#include "eteq/constant.hpp"
#include "eteq/functor.hpp"

#ifndef ETEQ_SESSION_HPP
#define ETEQ_SESSION_HPP

namespace eteq
{

using TensSetT = std::unordered_set<teq::iTensor*>;

struct iSession
{
	virtual ~iSession (void) = default;

	virtual void track (teq::TensT roots) = 0;

	/// update all nodes related to updated, if updated set is empty
	/// update all nodes related to the leaves (so everyone)
	/// ignore all nodes dependent on ignores including the ignored nodes
	virtual void update (TensSetT updated = {}, TensSetT ignores = {}) = 0;

	virtual void update_target (TensSetT target,
		TensSetT updated = {},
		TensSetT ignores = {}) = 0;
};

struct SizeT final
{
	size_t d = 0;

	operator size_t() const { return d; }
};

// for each leaf node, iteratively update the parents
// don't update parent node if it is part of ignored set
struct Session final : public iSession
{
	void track (teq::TensT roots) override
	{
		tracked_.insert(roots.begin(), roots.end());
		teq::ParentFinder pfinder;
		for (teq::TensptrT& root : roots)
		{
			root->accept(pfinder);
			root->accept(stat_);
		}
		auto& statmap = stat_.graphsize_;

		std::list<teq::iOperableFunc*> all_ops;
		for (auto& statpair : statmap)
		{
			if (0 < statpair.second.upper_)
			{
				// ensure we only track operable functors
				auto op = dynamic_cast<teq::iOperableFunc*>(statpair.first);
				if (nullptr == op)
				{
					logs::fatalf("cannot track non-operable functor %s",
						statpair.first->to_string().c_str());
				}
				all_ops.push_back(op);
			}
		}
		all_ops.sort(
			[&statmap](teq::iOperableFunc* a, teq::iOperableFunc* b)
			{
				return statmap[a].upper_ < statmap[b].upper_;
			});
		requirements_.clear();
		for (teq::iOperableFunc* op : all_ops)
		{
			auto& args = op->get_children();
			TensSetT unique_children;
			for (const teq::FuncArg& arg : args)
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
					static_cast<teq::iOperableFunc*>(parent_pair.first));
			}
		}
	}

	// this function is expected to be called repeatedly during runtime
	void update (TensSetT updated = {}, TensSetT ignores = {}) override
	{
		std::unordered_map<teq::iOperableFunc*,SizeT> fulfilments;
		for (teq::iTensor* unodes : updated)
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

	// this function is expected to be called repeatedly during runtime
	void update_target (TensSetT target, TensSetT updated = {},
		TensSetT ignores = {}) override
	{
		teq::OnceTraveler targetted;
		for (auto& tens : target)
		{
			tens->accept(targetted);
		}
		std::unordered_map<teq::iOperableFunc*,SizeT> fulfilments;
		updated.insert(ignores.begin(), ignores.end());
		for (teq::iTensor* unodes : updated)
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
				fulfilments[op.first].d >= op.second &&
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

	void optimize (const opt::OptCtx& rules)
	{
		teq::TensT tracked(tracked_.begin(), tracked_.end());
		opt::optimize(tracked, rules);
		stat_.graphsize_.clear();
		parents_.clear();
		track(tracked);
	}

	std::unordered_set<teq::TensptrT> tracked_;

	teq::GraphStat stat_;

	std::unordered_map<teq::iTensor*,
		std::unordered_set<teq::iOperableFunc*>> parents_;

	// List of operatible nodes and its number of unique children ordered from leaf to root
	std::vector<std::pair<teq::iOperableFunc*,size_t>> requirements_; // todo: test minimal requirements
};

}

#endif // ETEQ_SESSION_HPP
