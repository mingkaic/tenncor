#include "ade/tensor.hpp"
#include "ade/traveler.hpp"
#include "ade/functor.hpp"

#ifndef AGE_GRADER_HPP
#define AGE_GRADER_HPP

namespace age
{

using TensT = std::vector<ade::Tensorptr>;

struct iRuleSet
{
	virtual ~iRuleSet (void) = default;

	virtual ade::Tensor* data (double scalar, ade::Shape shape) = 0;

	virtual ade::Opcode sum_opcode (void) = 0;

	virtual ade::Opcode prod_opcode (void) = 0;

	virtual ade::Tensorptr grad_rule (size_t code, TensT args, size_t idx) = 0;
};

struct Grader final : public ade::iTraveler
{
	// this wouldn't be initialized in runtime library
	// (generator would initialize its custom ruleset)
	static std::unique_ptr<iRuleSet> rules_;

	Grader (const ade::iTensor* target) : target_(target) {}

	/// Implementation of iTraveler
	void visit (ade::Tensor* leaf) override
	{
		if (rules_ == nullptr)
		{
			err::fatal("cannot derive without ruleset");
		}
		if (leaf == target_)
		{
			derivatives_.emplace(leaf,
				rules_->data(1, target_->shape()));
		}
		else
		{
			derivatives_.emplace(leaf,
				rules_->data(0, target_->shape()));
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override;

	/// Target of tensor all visited nodes are derived with respect to
	const ade::iTensor* target_;

	std::unordered_map<const ade::iTensor*,ade::Tensorptr> derivatives_;
};

ade::ArgsT to_args (TensT tens);

ade::Tensorptr derive (ade::Tensorptr& root, const ade::iTensor* wrt);

}

#endif // AGE_GRADER_HPP
