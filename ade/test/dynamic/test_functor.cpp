
#ifndef DISABLE_FUNCTOR_TEST


#include "gtest/gtest.h"

#include "ade/functor.hpp"

#include "testutil/common.hpp"


struct FUNCTOR : public simple::TestModel {};


TEST_F(FUNCTOR, Childrens)
{
	simple::SessionT sess = get_session("FUNCTOR::Childrens");

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape());
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape());

	ade::OPCODE unary = (ade::OPCODE) sess->get_scalar("unary_op",
		{ade::ABS, ade::ROUND});
	ade::OPCODE binary = (ade::OPCODE) sess->get_scalar("binary_op",
		{ade::POW, ade::RAND_NORM});

	ade::Tensorptr fu = ade::Functor::get(unary, {{ade::identity, leaf}});
	ade::Tensorptr fb = ade::Functor::get(binary, {
		{ade::identity, leaf}, {ade::identity, leaf1}});

	ASSERT_NE(nullptr, fu.get());
	ASSERT_NE(nullptr, fb.get());

	ade::ArgsT uref = static_cast<ade::iFunctor*>(fu.get())->get_children();
	ade::ArgsT brefs = static_cast<ade::iFunctor*>(fb.get())->get_children();

	ASSERT_EQ(1, uref.size());
	EXPECT_EQ(leaf.get(), uref[0].second.get());
	ASSERT_EQ(2, brefs.size());
	EXPECT_EQ(leaf.get(), brefs[0].second.get());
	EXPECT_EQ(leaf1.get(), brefs[1].second.get());
}


TEST_F(FUNCTOR, ToString)
{
	simple::SessionT sess = get_session("FUNCTOR::ToString");

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape());
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape());

	ade::OPCODE unary = (ade::OPCODE) sess->get_scalar("unary_op",
		{ade::ABS, ade::ROUND});
	ade::OPCODE binary = (ade::OPCODE) sess->get_scalar("binary_op",
		{ade::POW, ade::RAND_NORM});

	ade::Tensorptr fu = ade::Functor::get(unary, {{ade::identity, leaf}});
	ade::Tensorptr fb = ade::Functor::get(binary, {
		{ade::identity, leaf}, {ade::identity, leaf1}});

	ASSERT_NE(nullptr, fu.get());
	ASSERT_NE(nullptr, fb.get());

	std::string out_unary = ade::opname(unary);
	EXPECT_STREQ(out_unary.c_str(), fu->to_string().c_str());

	std::string out_binary = ade::opname(binary);
	EXPECT_STREQ(out_binary.c_str(), fb->to_string().c_str());
}


TEST_F(FUNCTOR, OpNaming)
{
	simple::SessionT sess = get_session("FUNCTOR::OpNaming");

	ade::OPCODE op = (ade::OPCODE) sess->get_scalar("unary_op",
		{ade::ABS, ade::RAND_NORM});

	std::string name = ade::opname(op);
	EXPECT_EQ(op, ade::name_op(name));
}


#endif // DISABLE_FUNCTOR_TEST
