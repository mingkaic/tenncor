#include "gtest/gtest.h"

#include "ade/functor.hpp"

#include "testutil/common.hpp"


#ifndef DISABLE_FUNCTOR_TEST


struct FUNCTOR : public simple::TestModel {};


TEST_F(FUNCTOR, Gradient)
{
	simple::SessionT sess = get_session("FUNCTOR::Gradient");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Shape shape(slist);
	ade::Tensorptr badleaf = ade::Tensor::get(shape);

	ade::Tensorptr leaf = ade::Tensor::get(shape);
	ade::Tensorptr leaf1 = ade::Tensor::get(shape);

	ade::Tensorptr f = ade::Functor<ade::ADD>::get({leaf, leaf1});

	ASSERT_NE(nullptr, f.get());

	ade::Tensorptr gotwun = f->gradient(f);
	ade::Tensorptr got1p0 = f->gradient(leaf);
	ade::Tensorptr got0p1 = f->gradient(leaf1);
	ade::Tensorptr got0p0 = f->gradient(badleaf);

	std::string expectlabel = opname(ade::EXTEND) + "<" + shape.to_string() + ">";
	{
		auto wunrp = dynamic_cast<ade::Functor<ade::EXTEND,
			std::vector<ade::DimT>>*>(gotwun.get());
		ASSERT_NE(nullptr, wunrp);

		EXPECT_STREQ(expectlabel.c_str(), wunrp->to_string().c_str());
		std::vector<ade::iTensor*> wun_vec = wunrp->get_children();
		ASSERT_EQ(1, wun_vec.size());
		EXPECT_EQ(ade::Tensor::SYMBOLIC_ONE.get(), wun_vec[0]);
	}

	auto p10 = dynamic_cast<ade::Functor<ade::EXTEND,std::vector<ade::DimT>>*>(got1p0.get());
	auto p01 = dynamic_cast<ade::Functor<ade::EXTEND,std::vector<ade::DimT>>*>(got0p1.get());
	ASSERT_NE(nullptr, p10);
	ASSERT_NE(nullptr, p01);
	EXPECT_EQ(ade::Tensor::SYMBOLIC_ZERO.get(), got0p0.get());

	auto args10 = p10->get_children();
	auto args01 = p01->get_children();
	ASSERT_EQ(1, args10.size());
	ASSERT_EQ(1, args01.size());

	EXPECT_EQ(ade::Tensor::SYMBOLIC_ONE.get(), args10[0]);
	EXPECT_EQ(ade::Tensor::SYMBOLIC_ONE.get(), args01[0]);
}


TEST_F(FUNCTOR, Childrens)
{
	simple::SessionT sess = get_session("FUNCTOR::Childrens");

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape());
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape());

	ade::OPCODE unary = (ade::OPCODE) sess->get_scalar("unary_op", {ade::ABS, ade::FLIP});
	ade::OPCODE binary = (ade::OPCODE) sess->get_scalar("binary_op", {ade::POW, ade::RAND_NORM});

	ade::Tensorptr fu = ade::runtime_functor(unary, {leaf});
	ade::Tensorptr fb = ade::runtime_functor(binary, {leaf, leaf1});

	ASSERT_NE(nullptr, fu.get());
	ASSERT_NE(nullptr, fb.get());

	std::vector<ade::iTensor*> ref =
		static_cast<ade::iFunctor*>(fu.get())->get_children();
	std::vector<ade::iTensor*> refs =
		static_cast<ade::iFunctor*>(fb.get())->get_children();

	ASSERT_EQ(1, ref.size());
	EXPECT_EQ(leaf.get(), ref[0]);
	ASSERT_EQ(2, refs.size());
	EXPECT_EQ(leaf.get(), refs[0]);
	EXPECT_EQ(leaf1.get(), refs[1]);
}


TEST_F(FUNCTOR, ToString)
{
	simple::SessionT sess = get_session("FUNCTOR::ToString");

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape());
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape());

	ade::OPCODE unary = (ade::OPCODE) sess->get_scalar("unary_op", {ade::ABS, ade::FLIP});
	ade::OPCODE binary = (ade::OPCODE) sess->get_scalar("binary_op", {ade::POW, ade::RAND_NORM});

	ade::Tensorptr fu = ade::runtime_functor(unary, {leaf});
	ade::Tensorptr fb = ade::runtime_functor(binary, {leaf, leaf1});

	ASSERT_NE(nullptr, fu.get());
	ASSERT_NE(nullptr, fb.get());

	std::string out_unary = ade::opname(unary) + "<>";
	EXPECT_STREQ(out_unary.c_str(), fu->to_string().c_str());

	std::string out_binary = ade::opname(binary) + "<>";
	EXPECT_STREQ(out_binary.c_str(), fb->to_string().c_str());
}


#endif // DISABLE_FUNCTOR_TEST
