#include "teq/isession.hpp"

#include "eigen/eigen.hpp"

#include "distrib/reference.hpp"

#ifndef DISTRIB_ISESSION_HPP
#define DISTRIB_ISESSION_HPP

namespace distrib
{

using DRefMapsT = std::unordered_map<teq::iFunctor*,DRefsT>;

struct iDistribSess : public teq::iSession
{
	virtual ~iDistribSess (void) = default;

	/// Make requests to downstream for referenced data
	virtual void call (const DRefsT& refs) = 0;

	/// Wait for responses to all calls
	virtual void sync (void) = 0;

	virtual std::optional<std::string> lookup_id (teq::TensptrT tens) const = 0;

	virtual teq::TensptrT lookup_node (const std::string& id, bool recursive = true) = 0;

	virtual std::string get_id (void) const = 0;

	/// Implementation of iSession
	void track (const teq::TensptrSetT& roots) override
	{
		teq::TensptrSetT tracked;
		track_helper(tracked, roots);
		store_tracked(tracked);
	}

	/// Implementation of iSession
	void update (teq::iDevice& device,
		const teq::TensSetT& ignored = {}) override
	{
		teq::TensSetT rtens;
		rtens.reserve(roots_.size());
		std::transform(roots_.begin(), roots_.end(),
			std::inserter(rtens, rtens.end()),
			[](teq::TensptrT tens) { return tens.get(); });
		update_target(device, rtens, ignored);
	}

	/// Implementation of iSession
	void update_target (teq::iDevice& device,
		const teq::TensSetT& target,
		const teq::TensSetT& ignored = {}) override
	{
		teq::FuncListT reqs;
		teq::TensSetT nexts;
		std::unordered_set<teq::iDeviceRef*> devices;
		bool called = false;
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
					teq::iDeviceRef* ref = &op->device();
					assert(false == estd::has(devices, ref));
					devices.emplace(ref);

					reqs.push_front(op);
					if (estd::has(dependencies_, op))
					{
						call(dependencies_[op]);
						called = true;
					}
					auto children = op->get_dependencies();
					for (teq::TensptrT child : children)
					{
						nexts.emplace(child.get());
					}
				}
			}
		}
		if (called)
		{
			sync();
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
		dependencies_.clear();
	}

	/// Operable functors ordered by height in the tracked graph
	std::vector<teq::FuncSetT> ops_;

	const DRefMapsT& get_dependencies (void) const
	{
		return dependencies_;
	}

protected:
	/// Store all local nodes for remote referencing, by associating with a uuid
	virtual void store_tracked (const teq::TensptrSetT& locals) = 0;

	virtual void process_reqs (teq::FuncListT& reqs) {}

	void track_helper (teq::TensptrSetT& tracked, const teq::TensptrSetT& roots)
	{
		tracked.insert(roots.begin(), roots.end());
		teq::FuncsT funcs;
		teq::TensptrSetT nexts;
		for (auto root : roots)
		{
			if (false == estd::has(opheight_, root.get()))
			{
				if (auto f = dynamic_cast<teq::iFunctor*>(root.get()))
				{
					roots_.emplace(root);
					funcs.push_back(f);
					auto deps = f->get_dependencies();
					for (auto dep : deps)
					{
						if (auto ref = std::dynamic_pointer_cast<iDistRef>(dep))
						{
							dependencies_[f].push_back(ref);
						}
						else
						{
							nexts.emplace(dep);
						}
					}
				}
			}
		}
		if (funcs.empty())
		{
			return;
		}
		track_helper(tracked, nexts);
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
				ops_.insert(ops_.end(), maxheight - ops_.size(), teq::FuncSetT{});
			}
			ops_[maxheight - 1].emplace(func);
			opheight_.emplace(func, maxheight);
		}
	}

	teq::TensptrSetT roots_; // only needed for update

private:
	teq::TensMapT<size_t> opheight_;

	DRefMapsT dependencies_;
};

using DSessptrT = std::shared_ptr<iDistribSess>;

}

#endif // DISTRIB_ISESSION_HPP
