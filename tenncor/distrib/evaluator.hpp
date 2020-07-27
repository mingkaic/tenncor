
#include "distrib/imanager.hpp"

#ifndef DISTRIB_EVALUATOR_HPP
#define DISTRIB_EVALUATOR_HPP

namespace distr
{

struct DistEvaluator final : public teq::iEvaluator
{
	DistEvaluator (iDistManager* mgr) : mgr_(mgr) {}

	/// Implementation of iEvaluator
	void evaluate (
		teq::iDevice& device,
		const teq::TensSetT& targets,
		const teq::TensSetT& ignored = {}) override
	{
		// find all reachable refs and make remote call
		auto refs = reachable_refs(targets);
		estd::StrMapT<estd::StrSetT> deps;
		for (auto ref : refs)
		{
			deps[ref->cluster_id()].emplace(ref->node_id());
		}
		std::vector<std::future<void>> completions;
		for (auto& dpair : deps)
		{
			completions.push_back(mgr_->remote_evaluate(
				dpair.first, dpair.second));
		}
		// wait for completion before evaluating in local
		for (auto& done : completions)
		{
			while (done.valid() && done.wait_for(
				std::chrono::milliseconds(1)) ==
				std::future_status::timeout);
		}
		// locally evaluate
		teq::TravEvaluator eval(device, ignored);
		teq::multi_visit(eval, targets);
	}

	iDistManager* mgr_;
};

}

#endif // DISTRIB_EVALUATOR_HPP
