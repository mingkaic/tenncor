#include "opt/mock/target.hpp"

#ifdef OPT_MOCK_TARGET_HPP

opt::TargptrT build_mock_target (::TreeNode* target)
{
	opt::TargptrT out;
	switch (target->type_)
	{
		case ::TreeNode::ANY:
			out = std::make_shared<MockAny>(std::string(target->val_.any_));
			break;
		case ::TreeNode::SCALAR:
			out = std::make_shared<MockCst>(target->val_.scalar_);
			break;
		case ::TreeNode::FUNCTOR:
		{
			::Functor* func = target->val_.functor_;
			std::vector<opt::TargptrT> args;
			for (auto it = func->args_.head_; it != nullptr; it = it->next_)
			{
				args.push_back(build_mock_target((::TreeNode*) it->val_));
			}
			out = std::make_shared<MockFTarget>(std::string(func->name_),
				args, std::string(func->variadic_));
		}
			break;
		default:
			logs::fatalf("building unknown target %d", target->type_);
	}
	return out;
}

#endif // OPT_MOCK_TARGET_HPP
