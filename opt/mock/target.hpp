
#include "teq/mock/leaf.hpp"

#include "opt/target.hpp"

#ifndef OPT_MOCK_TARGET_HPP
#define OPT_MOCK_TARGET_HPP

const std::string tfactory_delim = ":";

struct MockTarget final : public opt::iTarget
{
    MockTarget (teq::TensptrT tag, const opt::TargptrsT& targs = {}) :
        tag_(tag), targs_(targs) {}

    teq::TensptrT convert (const query::SymbMapT& candidates) const override
    {
        return tag_;
    }

    teq::TensptrT tag_;

    opt::TargptrsT targs_;
};

struct MockTargetFactory final : public opt::iTargetFactory
{
    MockTargetFactory (void) : index_(0) {}

	opt::TargptrT make_scalar (double scalar,
		std::string sshape) const override
	{
		return std::make_shared<MockTarget>(
            std::make_shared<MockLeaf>(teq::Shape(),
            fmts::to_string(scalar) +
            tfactory_delim + sshape +
            tfactory_delim + fmts::to_string(index_++)));
	}

	opt::TargptrT make_symbol (std::string symbol) const override
	{
		return std::make_shared<MockTarget>(
            std::make_shared<MockLeaf>(teq::Shape(),
            symbol + tfactory_delim + fmts::to_string(index_++)));
	}

	opt::TargptrT make_functor (std::string opname,
		const google::protobuf::Map<std::string,query::Attribute>& attrs,
		const opt::TargptrsT& args) const override
	{
		return std::make_shared<MockTarget>(
            std::make_shared<MockLeaf>(teq::Shape(),
            opname + tfactory_delim + fmts::to_string(index_++)), args);
	}

    mutable size_t index_;
};

#endif // OPT_MOCK_TARGET_HPP
