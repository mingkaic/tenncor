#include "ade/tensor.hpp"
#include "ade/traveler.hpp"
#include "ade/functor.hpp"

#ifndef AGE_GRADER_HPP
#define AGE_GRADER_HPP

namespace age
{

using TensT = std::vector<ade::Tensorptr>;

// get these implemented
template <typename T>
ade::Tensor* data (T scalar, ade::Shape shape);

ade::Opcode sum_opcode (void);

ade::Opcode prod_opcode (void);

ade::Tensorptr grad_rule (size_t code, TensT args, size_t idx);

// already implemented
ade::ArgsT to_args (TensT tens);

struct Grader final : public ade::iTraveler
{
	Grader (const ade::iTensor* target) : target_(target) {}

	/// Implementation of iTraveler
	void visit (ade::Tensor* leaf) override
	{
		if (leaf == target_)
		{
			derivatives_.emplace(leaf,
                data(1, target_->shape()));
		}
		else
		{
			derivatives_.emplace(leaf,
                data(0, target_->shape()));
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override;

	/// Target of tensor all visited nodes are derived with respect to
	const ade::iTensor* target_;

    std::unordered_map<const ade::iTensor*,ade::Tensorptr> derivatives_;
};

}

#endif // AGE_GRADER_HPP
