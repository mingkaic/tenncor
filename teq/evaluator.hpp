
#include "teq/ievaluator.hpp"
#include "teq/traveler.hpp"

#ifndef TEQ_EVALUATOR_HPP
#define TEQ_EVALUATOR_HPP

namespace teq
{

struct TravEvaluator final : public iOnceTraveler
{
	TravEvaluator (iDevice& device, const TensSetT& ignored = {}) :
		ignored_(ignored), device_(&device) {}

	TensSetT ignored_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (iLeaf& leaf) override {}

	/// Implementation of iOnceTraveler
	void visit_func (iFunctor& func) override
	{
		if (estd::has(ignored_, &func))
		{
			return;
		}
		auto dependencies = func.get_dependencies();
		for (TensptrT dep : dependencies)
		{
			dep->accept(*this);
		}
		device_->calc(func);
	}

	iDevice* device_;
};

struct Evaluator final : public iEvaluator
{
	/// Implementation of iEvaluator
	void evaluate (
		iDevice& device,
		const TensSetT& targets,
		const TensSetT& ignored = {})
	{
		TravEvaluator eval(device, ignored);
		multi_visit(eval, targets);
	}
};

using EvalptrT = std::shared_ptr<Evaluator>;

}

#endif // TEQ_EVALUATOR_HPP
