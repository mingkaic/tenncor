
#include <future>

#include "distrib/reference.hpp"

#ifndef DISTRIB_ISESSION_HPP
#define DISTRIB_ISESSION_HPP

namespace estd
{

template <typename T>
using StrMapT = std::unordered_map<std::string,T>;

}

namespace distr
{

struct iDistribSess : public teq::iSession
{
	virtual ~iDistribSess (void) = default;

	virtual std::optional<std::string> lookup_id (teq::TensptrT tens) const = 0;

	virtual teq::TensptrT lookup_node (error::ErrptrT& err,
		const std::string& id, bool recursive = true) = 0;

	virtual std::string get_id (void) const = 0;

	/// Implementation of iSession
	void track (const teq::TensptrSetT& roots) override
	{
		teq::TensptrSetT tracked;
		track_helper(tracked, roots);
		store_tracked(tracked);
	}

	/// Implementation of iSession
	error::ErrptrT update_target (
		teq::iDevice& device,
		const teq::TensSetT& targets,
		const teq::TensSetT& ignored = {}) override
	{
		// check all targets are in tracked opheight_
		if (false == std::all_of(targets.begin(), targets.end(),
			[this](teq::iTensor* target)
			{
				return estd::has(opheight_, target);
			}))
		{
			return error::error("not all targets are tracked");
		}

		teq::FuncListT reqs;
		teq::TensSetT nexts;
		std::unordered_set<teq::iDeviceRef*> devices;
		std::unordered_map<std::string,std::unordered_set<std::string>> deps;
		// ignored tensors will never populate reqs
		for (auto rit = ops_.rbegin(), ret = ops_.rend();
			rit != ret; ++rit)
		{
			auto& fops = *rit;
			for (auto& op : fops)
			{
				if ((estd::has(nexts, op) || estd::has(targets, op)) &&
					false == estd::has(ignored, op))
				{
					teq::iDeviceRef* ref = &op->device();
					assert(false == estd::has(devices, ref));
					devices.emplace(ref);

					if (estd::has(dependencies_, op))
					{
						auto dep = dependencies_[op];
						auto cid = dep->cluster_id();
						deps[cid].emplace(dep->node_id());
					}
					else
					{
						reqs.push_front(op);
						auto children = op->get_dependencies();
						for (teq::TensptrT child : children)
						{
							nexts.emplace(child.get());
						}
					}
				}
			}
		}
		std::vector<std::future<void>> completions;
		for (auto& dpair : deps)
		{
			completions.push_back(call(dpair.first, dpair.second));
		}
		for (auto& done : completions)
		{
			while (done.valid() && done.wait_for(std::chrono::milliseconds(1)) ==
				std::future_status::timeout);
		}
		for (auto& op : reqs)
		{
			device.calc(*op);
		}
		process_reqs(reqs);
		return nullptr;
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

	DRefptrSetT get_dependencies (void) const
	{
		DRefptrSetT refs;
		refs.reserve(dependencies_.size());
		for (auto& dep : dependencies_)
		{
			refs.emplace(dep.second);
		}
		return refs;
	}

protected:
	virtual void process_reqs (teq::FuncListT& reqs) {}

	/// Store all local nodes for remote referencing, by associating with a uuid
	virtual void store_tracked (const teq::TensptrSetT& locals) = 0;

	/// Make requests to downstream for referenced data
	virtual std::future<void> call (
		const std::string& cluster_id,
		const std::unordered_set<std::string>& node_id) = 0;

private:
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
					funcs.push_back(f);
					if (auto ref = std::dynamic_pointer_cast<iDistRef>(root))
					{
						dependencies_.emplace(f, ref);
					}
					else
					{
						auto deps = f->get_dependencies();
						for (auto dep : deps)
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

	teq::TensMapT<size_t> opheight_;

	std::unordered_map<teq::iFunctor*,DRefptrT> dependencies_;
};

using DSessptrT = std::shared_ptr<iDistribSess>;

}

#endif // DISTRIB_ISESSION_HPP
