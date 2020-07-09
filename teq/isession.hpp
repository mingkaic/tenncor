///
/// session.hpp
/// teq
///
/// Purpose:
/// Define and implement session that tracks subgraphs
/// to allow rapidly update the tracked nodes
///

#include <cassert>
#include <list>

#include "teq/traveler.hpp"

#ifndef TEQ_ISESSION_HPP
#define TEQ_ISESSION_HPP

namespace teq
{

struct iDevice
{
	virtual ~iDevice (void) = default;

	virtual void calc (iTensor& tens) = 0;
};

/// Session interface that tracks and rapidly updates subgraphs
struct iSession
{
	virtual ~iSession (void) = default;

	/// Record subgraphs of roots
	virtual void track (const TensptrSetT& roots) = 0;

	/// Update every node under the subgraph except
	/// for the subgraphs of ignored
	/// this function is expected to be called repeatedly during runtime
	virtual void update (iDevice& device, const TensSetT& ignored = {}) = 0;

	/// Update every node under the target roots that are expected to be
	/// under the tracked subgraphs ignoring the subgraphs of ignored
	/// this function is expected to be called repeatedly during runtime
	virtual void update_target (iDevice& device,
		const TensSetT& target,
		const TensSetT& ignored = {}) = 0;

	/// Clear all tracked root and subgraph information
	virtual void clear (void) = 0;
};

using FuncListT = std::list<iFunctor*>;

/// iSession implementation that tracks subgraphs by ordering operable functors
/// in a vector such that parents are visited after children
struct Session : public iSession
{
	virtual ~Session (void) = default;

	/// Implementation of iSession
	void track (const TensptrSetT& roots) override
	{
		// filter for unvisited funcs from roots
		FuncsT funcs;
		TensptrSetT nexts;
		funcs.reserve(roots.size());
		for (auto root : roots)
		{
			if (false == estd::has(opheight_, root.get()))
			{
				if (auto f = dynamic_cast<iFunctor*>(root.get()))
				{
					roots_.emplace(root);
					funcs.push_back(f);
					auto children = f->get_dependencies();
					for (auto child : children)
					{
						nexts.emplace(child);
					}
				}
			}
		}
		// recursively calculate where to place funcs based on height
		if (funcs.empty())
		{
			return;
		}
		track(nexts);
		for (auto func : funcs)
		{
			size_t maxheight = 0;
			auto children = func->get_dependencies();
			for (auto child : children)
			{
				size_t height = estd::try_get(
					opheight_, child.get(), 0);
				maxheight = std::max(height, maxheight);
				roots_.erase(child);
			}
			maxheight += 1;
			if (maxheight > ops_.size())
			{
				ops_.insert(ops_.end(), maxheight - ops_.size(), FuncSetT{});
			}
			ops_[maxheight - 1].emplace(func);
			opheight_.emplace(func, maxheight);
		}
	}

	/// Implementation of iSession
	void update (iDevice& device, const TensSetT& ignored = {}) override
	{
		TensSetT rtens;
		rtens.reserve(roots_.size());
		std::transform(roots_.begin(), roots_.end(),
			std::inserter(rtens, rtens.end()),
			[](TensptrT tens) { return tens.get(); });
		update_target(device, rtens, ignored);
	}

	/// Implementation of iSession
	void update_target (iDevice& device,
		const TensSetT& target,
		const TensSetT& ignored = {}) override
	{
		FuncListT reqs;
		TensSetT nexts;
		std::unordered_set<iDeviceRef*> devices;
		// ignored tensors will never populate reqs
		for (auto rit = ops_.rbegin(), ret = ops_.rend();
			rit != ret; ++rit)
		{
			auto& fops = *rit;
			for (auto& op : fops)
			{
				if ((estd::has(nexts, op) || estd::has(target, op)) &&
					false == estd::has(ignored, op))
				{
					iDeviceRef* ref = &op->device();
					assert(false == estd::has(devices, ref));
					devices.emplace(ref);

					reqs.push_front(op);
					auto children = op->get_dependencies();
					for (TensptrT child : children)
					{
						nexts.emplace(child.get());
					}
				}
			}
		}

		for (auto& op : reqs)
		{
			device.calc(*op);
		}
		process_reqs(reqs);
	}

	/// Implementation of iSession
	void clear (void) override
	{
		ops_.clear();
		opheight_.clear();
	}

	/// Operable functors ordered by height in the tracked graph
	std::vector<FuncSetT> ops_;

protected:
	virtual void process_reqs (FuncListT& reqs) {}

	TensptrSetT roots_; // only needed for update

private:
	TensMapT<size_t> opheight_;
};

}

#endif // TEQ_ISESSION_HPP
