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

	/// update all nodes related to the leaves (so everyone)
	/// ignore all nodes dependent on ignored including the ignored nodes
	virtual void update (TensSetT ignored = {}) = 0;

	virtual void update_target (TensSetT target, TensSetT ignored = {}) = 0;
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
		ops_.clear();
		tracked_.insert(roots.begin(), roots.end());

		teq::GraphStat stat;
		teq::ParentFinder pfinder; // revert
		for (teq::TensptrT& root : roots)
		{
			root->accept(stat);
			root->accept(pfinder); // revert
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
			[&statmap, &pfinder](teq::iOperableFunc* a, teq::iOperableFunc* b)
			{
				if (statmap[a].upper_ == statmap[b].upper_)
				{
					size_t aupper = 0, bupper = 0;
					for (auto& pp : pfinder.parents_[a])
					{
						aupper = std::max(aupper, statmap[pp.first].upper_);
					}
					for (auto& pp : pfinder.parents_[b])
					{
						bupper = std::max(bupper, statmap[pp.first].upper_);
					}
					if (aupper == bupper)
					{
						return a->shape().n_elems() < b->shape().n_elems();
					}
					return aupper < bupper;
				}
				return statmap[a].upper_ < statmap[b].upper_;
			}); // todo: revert this back
	}

	// this function is expected to be called repeatedly during runtime
	void update (TensSetT ignored = {}) override
	{
		std::list<teq::iOperableFunc*> reqs;
		TensSetT acceptable;
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

	// this function is expected to be called repeatedly during runtime
	void update_target (TensSetT target, TensSetT ignored = {}) override
	{
		std::list<teq::iOperableFunc*> reqs;
		TensSetT acceptable;
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

	void optimize (const opt::OptCtx& rules)
	{
		teq::TensT tracked(tracked_.begin(), tracked_.end());
		opt::optimize(tracked, rules);
		track(tracked);
	}

	std::unordered_set<teq::TensptrT> tracked_;

	std::vector<teq::iOperableFunc*> ops_;
};

}

#endif // ETEQ_SESSION_HPP
