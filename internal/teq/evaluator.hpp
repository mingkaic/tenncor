
#ifndef TEQ_EVALUATOR_HPP
#define TEQ_EVALUATOR_HPP

#include "internal/teq/ievaluator.hpp"
#include "internal/teq/traveler.hpp"

namespace teq
{

struct TravEvaluator final : public iOnceTraveler
{
	TravEvaluator (iDevice& device,
		const TensSetT& targets, const TensSetT& ignored) :
		ignored_(ignored), device_(&device), targets_(targets)
	{
		for (auto ig : ignored)
		{
			if (nullptr != ig && nullptr == ig->device().data())
			{
				global::throw_errf("cannot ignore tensor %s without existing data",
					ig->to_string().c_str());
			}
		}
	}

	TensSetT ignored_;

private:
	/// Implementation of iOnceTraveler
	void visit_leaf (iLeaf&) override {}

	/// Implementation of iOnceTraveler
	void visit_func (iFunctor& func) override
	{
		if (estd::has(ignored_, &func))
		{
			return;
		}
		auto dependencies = func.get_args();
		multi_visit(*this, dependencies);
		device_->calc(func, (size_t) estd::has(targets_, &func));
	}

	iDevice* device_;

	TensSetT targets_;
};

struct Evaluator final : public iEvaluator
{
	/// Implementation of iEvaluator
	void evaluate (
		iDevice& device,
		const TensSetT& targets,
		const TensSetT& ignored = {}) override
	{
		TravEvaluator eval(device, targets, ignored);
		multi_visit(eval, targets);
	}
};

using EvalptrT = std::shared_ptr<Evaluator>;

void set_eval (iEvaluator* eval, global::CfgMapptrT ctx = global::context());

iEvaluator& get_eval (const global::CfgMapptrT& ctx = global::context());

}

#endif // TEQ_EVALUATOR_HPP
