
#include "distrib/manager.hpp"

#ifndef DISTRIB_OP_EVALUATOR_HPP
#define DISTRIB_OP_EVALUATOR_HPP

namespace distr
{

struct DistrEvaluator final : public teq::iEvaluator
{
	DistrEvaluator (iDistrManager& mgr) : svc_(distr::get_opsvc(mgr)) {}

	/// Implementation of iEvaluator
	void evaluate (
		teq::iDevice& device,
		const teq::TensSetT& targets,
		const teq::TensSetT& ignored = {}) override
	{
		svc_.evaluate(device, targets, ignored);
	}

	DistrOpService& svc_;
};

}

#endif // DISTRIB_OP_EVALUATOR_HPP
