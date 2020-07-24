
#include <future>

#include "error/error.hpp"

#include "distrib/reference.hpp"

#ifndef DISTRIB_IEVALUATOR_HPP
#define DISTRIB_IEVALUATOR_HPP

namespace distr
{

struct iDistrEvaluator : public teq::iEvaluator
{
	virtual ~iDistrEvaluator (void) = default;

	virtual void expose_node (teq::TensptrT tens) = 0;

	virtual std::optional<std::string> lookup_id (
		teq::TensptrT tens) const = 0;

	virtual teq::TensptrT lookup_node (error::ErrptrT& err,
		const std::string& id, bool recursive = true) = 0;

	virtual std::string get_id (void) const = 0;

	virtual DRefptrSetT get_remotes (void) const = 0;

	/// Implementation of iEvaluator
	void evaluate (
		teq::iDevice& device,
		const teq::TensSetT& targets,
		const teq::TensSetT& ignored = {})
	{
		// find all reachable refs and make remote call
		auto refs = reachable_refs(targets);
		estd::StrMapT<std::unordered_set<std::string>> deps;
		for (auto ref : refs)
		{
			deps[ref->cluster_id()].emplace(ref->node_id());
		}
		std::vector<std::future<void>> completions;
		for (auto& dpair : deps)
		{
			completions.push_back(call(dpair.first, dpair.second));
		}
		// wait for completion before evaluating in local
		for (auto& done : completions)
		{
			while (done.valid() && done.wait_for(std::chrono::milliseconds(1)) ==
				std::future_status::timeout);
		}
		// locally evaluate
		teq::TravEvaluator eval(device, ignored);
		multi_visit(eval, targets);
	}

protected:
	/// Make requests to downstream for referenced data
	virtual std::future<void> call (const std::string& cluster_id,
		const std::unordered_set<std::string>& node_id) = 0;
};

using iDEvalptrT = std::shared_ptr<iDistrEvaluator>;

}

#endif // DISTRIB_IEVALUATOR_HPP
