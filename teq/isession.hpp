///
/// session.hpp
/// teq
///
/// Purpose:
/// Define and implement session that tracks subgraphs
/// to allow rapidly update the tracked nodes
///

#include <list>

#include "teq/traveler.hpp"

#ifndef TEQ_ISESSION_HPP
#define TEQ_ISESSION_HPP

namespace teq
{

/// Session interface that tracks and rapidly updates subgraphs
struct iSession
{
	virtual ~iSession (void) = default;

	/// Record subgraphs of roots
	virtual void track (TensptrsT roots) = 0;

	/// Update every node under the subgraph except
	/// for the subgraphs of ignored
	/// this function is expected to be called repeatedly during runtime
	virtual void update (TensSetT ignored = {}) = 0;

	/// Update every node under the target roots that are expected to be
	/// under the tracked subgraphs ignoring the subgraphs of ignored
	/// this function is expected to be called repeatedly during runtime
	virtual void update_target (TensSetT target, TensSetT ignored = {}) = 0;

	/// Clear all tracked root and subgraph information
	virtual void clear (void) = 0;

	/// Return set of tracked tensor roots
	virtual TensptrSetT get_tracked (void) const = 0;
};

using FuncListT = std::list<iFunctor*>;

struct iDevice
{
	virtual ~iDevice (void) = default;

	virtual void calc (iTensor& tens) = 0;
};

// struct DynamicSession : public iSession
// {
// 	DynamicSession (iDevice& device) : device_(&device) {}

// 	virtual ~DynamicSession (void) = default;

// 	/// Implementation of iSession
// 	void track (TensptrsT roots) override
// 	{
// 		tracked_.insert(roots.begin(), roots.end());

// 		GraphStat stat;
// 		for (const TensptrT& root : tracked_)
// 		{
// 			root->accept(stat);
// 		}
// 		auto& statmap = stat.graphsize_;

// 		for (auto& statpair : statmap)
// 		{
// 			if (0 < statpair.second.upper_)
// 			{
// 				ops_.push_back(static_cast<iFunctor*>(statpair.first));
// 			}
// 		}
// 		std::sort(ops_.begin(), ops_.end(),
// 			[&statmap](iFunctor* a, iFunctor* b)
// 			{ return statmap[a].upper_ < statmap[b].upper_; });
// 	}

//     /// Implementation of iSession
//     void update (TensSetT ignored = {}) override
// 	{}

// 	/// Implementation of iSession
// 	void update_target (TensSetT target, TensSetT ignored = {}) override
// 	{}

// 	/// Implementation of iSession
// 	void clear (void) override
// 	{}

// 	/// Implementation of iSession
// 	TensptrSetT get_tracked (void) const override
// 	{
// 		return tracked_;
// 	}

// 	/// Set of all tensors input through tracked function
// 	/// The set of roots of all session graphs is a possible subset
// 	TensptrSetT tracked_;

// 	/// Operable functors ordered by height in the tracked graph
// 	std::vector<FuncSetT> ops_;

// private:
// 	iDevice* device_;
// };

/// iSession implementation that tracks subgraphs by ordering operable functors
/// in a vector such that parents are visited after children
struct Session : public iSession
{
	Session (iDevice& device) : device_(&device) {}

	virtual ~Session (void) = default;

	/// Implementation of iSession
	void track (TensptrsT roots) override
	{
		ops_.clear();
		tracked_.insert(roots.begin(), roots.end());

		GraphStat stat;
		for (const TensptrT& root : tracked_)
		{
			root->accept(stat);
		}
		auto& statmap = stat.graphsize_;

		for (auto& statpair : statmap)
		{
			if (0 < statpair.second.upper_)
			{
				ops_.push_back(static_cast<iFunctor*>(statpair.first));
			}
		}
		std::sort(ops_.begin(), ops_.end(),
			[&statmap](iFunctor* a, iFunctor* b)
			{ return statmap[a].upper_ < statmap[b].upper_; });
	}

	/// Implementation of iSession
	void update (TensSetT ignored = {}) override
	{
		FuncListT reqs;
		TensSetT acceptable;
		for (auto& root : tracked_)
		{
			acceptable.emplace(root.get());
		}
		std::unordered_set<teq::iDeviceRef*> devices;
		// ignored tensors will never populate reqs
		for (auto rit = ops_.rbegin(), ret = ops_.rend();
			rit != ret; ++rit)
		{
			auto& op = *rit;
			if (estd::has(acceptable, op) &&
				false == estd::has(ignored, op))
			{
				if (false == estd::has(devices, &op->device()))
				{
					reqs.push_front(op);
					devices.emplace(&op->device());
				}
				auto children = op->get_children();
				for (TensptrT child : children)
				{
					acceptable.emplace(child.get());
				}
			}
		}

		calc_reqfuncs(reqs);
	}

	/// Implementation of iSession
	void update_target (TensSetT target, TensSetT ignored = {}) override
	{
		FuncListT reqs;
		TensSetT acceptable;
		for (auto& root : target)
		{
			acceptable.emplace(root);
		}
		std::unordered_set<teq::iDeviceRef*> devices;
		// ignored tensors will never populate reqs
		for (auto rit = ops_.rbegin(), ret = ops_.rend();
			rit != ret; ++rit)
		{
			auto& op = *rit;
			if (estd::has(acceptable, op) &&
				false == estd::has(ignored, op))
			{
				if (false == estd::has(devices, &op->device()))
				{
					reqs.push_front(op);
					devices.emplace(&op->device());
				}
				auto children = op->get_children();
				for (TensptrT child : children)
				{
					acceptable.emplace(child.get());
				}
			}
		}

		calc_reqfuncs(reqs);
	}

	/// Implementation of iSession
	void clear (void) override
	{
		ops_.clear();
		tracked_.clear();
	}

	/// Implementation of iSession
	TensptrSetT get_tracked (void) const override
	{
		return tracked_;
	}

	/// Set of all tensors input through tracked function
	/// The set of roots of all session graphs is a possible subset
	TensptrSetT tracked_;

	/// Operable functors ordered by height in the tracked graph
	FuncsT ops_;

protected:
	virtual void process_reqs (FuncListT& reqs) {}

private:
	void calc_reqfuncs (FuncListT& reqs)
	{
		for (auto& op : reqs)
		{
			device_->calc(*op);
		}
		process_reqs(reqs);
	}

	iDevice* device_;
};

const std::string device_key = "device";

#define DEVICE_INIT(DEVICE_TYPE)::config::global_config.add_entry<DEVICE_TYPE>(teq::device_key)

}

#endif // TEQ_ISESSION_HPP
