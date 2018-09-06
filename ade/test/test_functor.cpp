#include "gtest/gtest.h"

#include "ade/test/common.hpp"

#include "ade/functor.hpp"


#ifndef DISABLE_FUNCTOR_TEST


TEST(FUNCTOR, Gradient)
{
	// SESSION sess = getSession("FUNCTOR::Gradient");

	// std::vector<ade::DimT> slist = get_shape(sess, "slist");
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr badleaf = ade::Tensor::get(ade::Shape(slist));

	// ade::OPCODE unary = sess->get_scalar("unary_op", {ade::ABS, ade::FLIP});
	// ade::OPCODE binary = sess->get_scalar("binary_op", {ade::POW, ade::NORM});

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr f = ade::Functor<ade::ADD>::get({leaf, leaf1});

	ASSERT_NE(nullptr, f.get());

	ade::Tensorptr gotwun = f->gradient(f);
	ade::Tensorptr got1p0 = f->gradient(leaf);
	ade::Tensorptr got0p1 = f->gradient(leaf1);
	ade::Tensorptr got0p0 = f->gradient(badleaf);

	std::string expectlabel = opname(ade::RESHAPE) + "<[2\\3]>";
	{
		auto wunrp = dynamic_cast<ade::Functor<ade::RESHAPE,
			std::vector<ade::DimT>>*>(gotwun.get());
		ASSERT_NE(nullptr, wunrp);

		EXPECT_STREQ(expectlabel.c_str(), wunrp->to_string().c_str());
		std::vector<ade::iTensor*> wun_vec = wunrp->get_refs();
		ASSERT_EQ(1, wun_vec.size());
		EXPECT_EQ(ade::Tensor::SYMBOLIC_ONE.get(), wun_vec[0]);
	}

	auto p10 = dynamic_cast<ade::Functor<ade::ADD>*>(got1p0.get());
	auto p01 = dynamic_cast<ade::Functor<ade::ADD>*>(got0p1.get());
	auto p00 = dynamic_cast<ade::Functor<ade::ADD>*>(got0p0.get());
	ASSERT_NE(nullptr, p10);
	ASSERT_NE(nullptr, p01);
	ASSERT_NE(nullptr, p00);

	auto args10 = p10->get_refs();
	auto args01 = p01->get_refs();
	auto args00 = p00->get_refs();
	ASSERT_EQ(2, args10.size());
	ASSERT_EQ(2, args01.size());
	ASSERT_EQ(2, args00.size());

	{
		auto wunrp = dynamic_cast<ade::Functor<ade::RESHAPE,
			std::vector<ade::DimT>>*>(args10[0]);
		auto zrorp = dynamic_cast<ade::Functor<ade::RESHAPE,
			std::vector<ade::DimT>>*>(args10[1]);
		ASSERT_NE(nullptr, wunrp);
		ASSERT_NE(nullptr, zrorp);

		EXPECT_STREQ(expectlabel.c_str(), wunrp->to_string().c_str());
		EXPECT_STREQ(expectlabel.c_str(), zrorp->to_string().c_str());
		std::vector<ade::iTensor*> wun_vec = wunrp->get_refs();
		std::vector<ade::iTensor*> zro_vec = zrorp->get_refs();
		ASSERT_EQ(1, wun_vec.size());
		ASSERT_EQ(1, zro_vec.size());
		EXPECT_EQ(ade::Tensor::SYMBOLIC_ONE.get(), wun_vec[0]);
		EXPECT_EQ(ade::Tensor::SYMBOLIC_ZERO.get(), zro_vec[0]);
	}

	{
		auto zrorp = dynamic_cast<ade::Functor<ade::RESHAPE,
			std::vector<ade::DimT>>*>(args01[0]);
		auto wunrp = dynamic_cast<ade::Functor<ade::RESHAPE,
			std::vector<ade::DimT>>*>(args01[1]);
		ASSERT_NE(nullptr, zrorp);
		ASSERT_NE(nullptr, wunrp);

		EXPECT_STREQ(expectlabel.c_str(), zrorp->to_string().c_str());
		EXPECT_STREQ(expectlabel.c_str(), wunrp->to_string().c_str());
		std::vector<ade::iTensor*> zro_vec = zrorp->get_refs();
		std::vector<ade::iTensor*> wun_vec = wunrp->get_refs();
		ASSERT_EQ(1, zro_vec.size());
		ASSERT_EQ(1, wun_vec.size());
		EXPECT_EQ(ade::Tensor::SYMBOLIC_ZERO.get(), zro_vec[0]);
		EXPECT_EQ(ade::Tensor::SYMBOLIC_ONE.get(), wun_vec[0]);
	}

	{
		auto zrorp = dynamic_cast<ade::Functor<ade::RESHAPE,
			std::vector<ade::DimT>>*>(args00[0]);
		auto zrorp2 = dynamic_cast<ade::Functor<ade::RESHAPE,
			std::vector<ade::DimT>>*>(args00[1]);
		ASSERT_NE(nullptr, zrorp);
		ASSERT_NE(nullptr, zrorp2);

		EXPECT_STREQ(expectlabel.c_str(), zrorp->to_string().c_str());
		EXPECT_STREQ(expectlabel.c_str(), zrorp2->to_string().c_str());
		std::vector<ade::iTensor*> zro_vec = zrorp->get_refs();
		std::vector<ade::iTensor*> zro_vec2 = zrorp2->get_refs();
		ASSERT_EQ(1, zro_vec.size());
		ASSERT_EQ(1, zro_vec2.size());
		EXPECT_EQ(ade::Tensor::SYMBOLIC_ZERO.get(), zro_vec[0]);
		EXPECT_EQ(ade::Tensor::SYMBOLIC_ZERO.get(), zro_vec2[0]);
	}
}


TEST(FUNCTOR, Refs)
{
	// SESSION sess = getSession("FUNCTOR::Refs");

	// std::vector<ade::DimT> slist = get_shape(sess, "slist");
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr f = ade::Functor<ade::ADD>::get({leaf, leaf1});

	ASSERT_NE(nullptr, f.get());

	std::vector<ade::iTensor*> refs =
		static_cast<ade::Functor<ade::ADD>*>(f.get())->get_refs();

	ASSERT_EQ(2, refs.size());
	EXPECT_EQ(leaf.get(), refs[0]);
	EXPECT_EQ(leaf1.get(), refs[1]);
}


TEST(FUNCTOR, ToString)
{
	// SESSION sess = getSession("FUNCTOR::ToString");

	// std::vector<ade::DimT> slist = get_shape(sess, "slist");
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr f = ade::Functor<ade::ADD>::get({leaf, leaf1});

	ASSERT_NE(nullptr, f.get());

	std::string expect_out = "ADD<>";
	EXPECT_STREQ(expect_out.c_str(), f->to_string().c_str());
}


#endif /* DISABLE_FUNCTOR_TEST */
