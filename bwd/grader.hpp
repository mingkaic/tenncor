///
///	grader.hpp
///	bwd
///
///	Purpose:
///	Define grader traveler to build partial derivative equations
///

#include "ade/tensor.hpp"
#include "ade/traveler.hpp"
#include "ade/functor.hpp"

#ifndef BWD_GRADER_HPP
#define BWD_GRADER_HPP

namespace age
{

/// Vector representation of tensor pointers
using TensT = std::vector<ade::Tensorptr>;

/// Ruleset used by a Grader traveler to derive equations
struct iRuleSet
{
	virtual ~iRuleSet (void) = default;

	/// Return tensor leaf containing scalar of specific shape
	virtual ade::Tensor* data (double scalar, ade::Shape shape) = 0;

	/// Return opcode representing nnary sum
	virtual ade::Opcode sum_opcode (void) = 0;

	/// Return opcode representing binary multiplication
	virtual ade::Opcode prod_opcode (void) = 0;

	/// Return chain rule of operation with respect to argument at idx
	/// specified by code given args
	virtual ade::Tensorptr grad_rule (size_t code, TensT args, size_t idx) = 0;
};

/// Traveler to obtain derivative of accepted node with respect to target
struct Grader final : public ade::iTraveler
{
	// this wouldn't be initialized in runtime library
	// (generator would initialize its custom ruleset)
	static std::shared_ptr<iRuleSet> default_rules;

	Grader (const ade::iTensor* target, std::shared_ptr<iRuleSet> rules = default_rules) :
		target_(target), rules_(rules)
	{
		if (target_ == nullptr)
		{
			err::fatal("cannot derive with respect to null");
		}
		if (rules_ == nullptr)
		{
			err::fatal("cannot derive without ruleset");
		}
	}

	/// Implementation of iTraveler
	void visit (ade::Tensor* leaf) override
	{
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

	/// Map forward root node to derivative root
	std::unordered_map<const ade::iTensor*,ade::Tensorptr> derivatives_;

private:
	/// Ruleset used by this instance
	std::shared_ptr<iRuleSet> rules_;
};

/// Return ArgsT with each tensor in TensT attached to identity mapper
ade::ArgsT to_args (TensT tens);

/// Return derivative of root with respect to wrt using Grader
ade::Tensorptr derive (ade::Tensorptr& root, const ade::iTensor* wrt);

}

#endif // BWD_GRADER_HPP