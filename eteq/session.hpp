///
/// session.hpp
/// eteq
///
/// Purpose:
/// Define and implement session that tracks subgraphs and
/// rapidly updates the tracked graph or a portion of tracked graph
///

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
struct iSession
{
	virtual ~iSession (void) = default;

	/// Record subgraphs of roots
	virtual void track (teq::TensptrsT roots) = 0;

	/// Update every node under the subgraph except
	/// for the subgraphs of ignored
	/// this function is expected to be called repeatedly during runtime
	virtual void update (teq::TensSetT ignored = {}) = 0;

	/// Update every node under the target roots that are expected to be
	/// under the tracked subgraphs ignoring the subgraphs of ignored
	/// this function is expected to be called repeatedly during runtime
	virtual void update_target (teq::TensSetT target, teq::TensSetT ignored = {}) = 0;
};

/// iSession implementation that tracks subgraphs by ordering operable functors
/// in a vector such that parents are visited after children
struct Session final : public iSession
{
	/// Implementation of iSession
	void track (teq::TensptrsT roots) override
	{
		ops_.clear();
		tracked_.insert(roots.begin(), roots.end());

		teq::GraphStat stat;
		for (teq::TensptrT& root : roots)
		{
			root->accept(stat);
		}
		auto& statmap = stat.graphsize_;

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
				ops_.push_back(op);
			}
		}
		std::sort(ops_.begin(), ops_.end(),
			[&statmap](teq::iOperableFunc* a, teq::iOperableFunc* b)
			{ return statmap[a].upper_ < statmap[b].upper_; });
	}

	/// Implementation of iSession
	void update (teq::TensSetT ignored = {}) override
	{
		std::list<teq::iOperableFunc*> reqs;
		teq::TensSetT acceptable;
		for (auto& root : tracked_)
		{
			acceptable.emplace(root.get());
		}
		// ignored tensors will never populate reqs
		for (auto rit = ops_.rbegin(), ret = ops_.rend();
			rit != ret; ++rit)
		{
			auto& op = *rit;
			if (estd::has(acceptable, op) &&
				false == estd::has(ignored, op))
			{
				reqs.push_front(op);
				auto& children = op->get_children();
				for (auto& child : children)
				{
					acceptable.emplace(child.get_tensor().get());
				}
			}
		}

		for (auto& op : reqs)
		{
			op->update();
		}
	}

	/// Implementation of iSession
	void update_target (teq::TensSetT target, teq::TensSetT ignored = {}) override
	{
		std::list<teq::iOperableFunc*> reqs;
		teq::TensSetT acceptable;
		for (auto& root : target)
		{
			acceptable.emplace(root);
		}
		// ignored tensors will never populate reqs
		for (auto rit = ops_.rbegin(), ret = ops_.rend();
			rit != ret; ++rit)
		{
			auto& op = *rit;
			if (estd::has(acceptable, op) &&
				false == estd::has(ignored, op))
			{
				reqs.push_front(op);
				auto& children = op->get_children();
				for (auto& child : children)
				{
					acceptable.emplace(child.get_tensor().get());
				}
			}
		}

		for (auto& op : reqs)
		{
			op->update();
		}
	}

	/// Apply input optimization rules using opt module, then re-track
	void optimize (const opt::OptCtx& rules)
	{
		teq::TensptrsT tracked(tracked_.begin(), tracked_.end());
		opt::optimize(tracked, rules);
		track(tracked);
	}

	/// Set of all tensors input through tracked function
	/// The set of roots of all session graphs is a possible subset
	teq::TensptrSetT tracked_;

	/// Operable functors ordered by height in the tracked graph
	std::vector<teq::iOperableFunc*> ops_;
};

}

#endif // ETEQ_SESSION_HPP
