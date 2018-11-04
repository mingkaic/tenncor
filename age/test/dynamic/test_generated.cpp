
#ifndef DISABLE_GENERATED_TEST


#include "gtest/gtest.h"

#include "age/grader.hpp"

#include "testutil/common.hpp"


struct GENERATED : public simple::TestModel {};


TEST_F(GENERATED, Childrens)
{
	simple::SessionT sess = get_session("GENERATED::Childrens");

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape());
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape());

	age::OPCODE unary = (age::OPCODE) sess->get_scalar("unary_op",
		{ade::ABS, ade::ROUND});
	age::OPCODE binary = (age::OPCODE) sess->get_scalar("binary_op",
		{ade::POW, ade::RAND_NORM});

	ade::Tensorptr fu = ade::Functor::get(make_code(unary),
		{{ade::identity, leaf}});
	ade::Tensorptr fb = ade::Functor::get(make_code(binary),
		{{ade::identity, leaf}, {ade::identity, leaf1}});

	ASSERT_NE(nullptr, fu.get());
	ASSERT_NE(nullptr, fb.get());

	ade::ArgsT uref = static_cast<ade::iFunctor*>(fu.get())->get_children();
	ade::ArgsT brefs = static_cast<ade::iFunctor*>(fb.get())->get_children();

	ASSERT_EQ(1, uref.size());
	EXPECT_EQ(leaf.get(), uref[0].tensor_.get());
	ASSERT_EQ(2, brefs.size());
	EXPECT_EQ(leaf.get(), brefs[0].tensor_.get());
	EXPECT_EQ(leaf1.get(), brefs[1].tensor_.get());
}


TEST_F(GENERATED, ToString)
{
	simple::SessionT sess = get_session("GENERATED::ToString");

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape());
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape());

	age::OPCODE unary = (age::OPCODE) sess->get_scalar("unary_op",
		{ade::ABS, ade::ROUND});
	age::OPCODE binary = (age::OPCODE) sess->get_scalar("binary_op",
		{ade::POW, ade::RAND_NORM});

	ade::Tensorptr fu = ade::Functor::get(make_code(unary),
		{{ade::identity, leaf}});
	ade::Tensorptr fb = ade::Functor::get(make_code(binary),
		{{ade::identity, leaf}, {ade::identity, leaf1}});

	ASSERT_NE(nullptr, fu.get());
	ASSERT_NE(nullptr, fb.get());

	std::string out_unary = age::opname(unary);
	EXPECT_STREQ(out_unary.c_str(), fu->to_string().c_str());

	std::string out_binary = age::opname(binary);
	EXPECT_STREQ(out_binary.c_str(), fb->to_string().c_str());
}


TEST_F(GENERATED, OpNaming)
{
	simple::SessionT sess = get_session("GENERATED::OpNaming");

	age::OPCODE op = (age::OPCODE) sess->get_scalar("unary_op",
		{ade::ABS, ade::RAND_NORM});

	std::string name = age::opname(op);
	EXPECT_EQ(op, ade::name_op(name));
}


#endif // DISABLE_GENERATED_TEST
