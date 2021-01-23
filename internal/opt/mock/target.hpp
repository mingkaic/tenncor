
#ifndef OPT_MOCK_TARGET_HPP
#define OPT_MOCK_TARGET_HPP

#include "internal/opt/opt.hpp"

#include "internal/teq/mock/mock.hpp"

#include "gmock/gmock.h"

struct MockTarget final : public opt::iTarget
{
	MOCK_CONST_METHOD1(convert, teq::TensptrT(const query::SymbMapT&));
};

struct MockTargetFactory final : public opt::iTargetFactory
{
	MOCK_CONST_METHOD2(make_scalar, opt::TargptrT(double,std::string));

	MOCK_CONST_METHOD1(make_symbol, opt::TargptrT(const std::string&));

	MOCK_CONST_METHOD3(make_functor, opt::TargptrT(const std::string&,
		const google::protobuf::Map<std::string,query::Attribute>&,const opt::TargptrsT&));
};

#endif // OPT_MOCK_TARGET_HPP
