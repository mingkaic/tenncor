#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "opt/optimize.hpp"

#ifndef OPT_MOCK_TARGET_HPP
#define OPT_MOCK_TARGET_HPP

struct MockAny final : public opt::iTarget
{
	MockAny (std::string symbol) : symbol_(symbol) {}

	teq::TensptrT convert (
		teq::Shape outshape, const opt::Candidate& candidate) const override
	{
		return estd::must_getf(candidate.anys_, symbol_,
			"cannot find any symbol %s", symbol_.c_str());
	}

	std::string symbol_;
};

struct MockCst final : public opt::iTarget
{
	MockCst (double scalar) : scalar_(scalar) {}

	teq::TensptrT convert (
		teq::Shape outshape, const opt::Candidate& candidate) const override
	{
		return std::make_shared<MockLeaf>(
			outshape, fmts::to_string(scalar_));
	}

	double scalar_;
};

struct MockFTarget final : public opt::iTarget
{
	MockFTarget (std::string opname, std::vector<opt::TargptrT> args,
		std::string variadic) :
		opname_(opname), args_(args), variadic_(variadic) {}

	teq::TensptrT convert (
		teq::Shape outshape, const opt::Candidate& candidate) const override
	{
		teq::TensptrsT args;
		for (auto& targ : args_)
		{
			args.push_back(targ->convert(outshape, candidate));
		}
		if (variadic_.size() > 0)
		{
			auto& edges = candidate.variadic_.at(variadic_);
			for (teq::TensptrT edge : edges)
			{
				args.push_back(edge);
			}
		}
		return std::make_shared<MockFunctor>(args, teq::Opcode{opname_, 0});
	}

	std::string opname_;

	std::vector<opt::TargptrT> args_;

	std::string variadic_;
};

opt::TargptrT build_mock_target (::TreeNode* target);

#endif // OPT_MOCK_TARGET_HPP
