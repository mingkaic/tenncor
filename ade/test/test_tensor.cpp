#include "gtest/gtest.h"

#include "ade/test/common.hpp"

#include "ade/tensor.hpp"
#include "ade/functor.hpp"


#ifndef DISABLE_TENSOR_TEST


TEST(TENSOR, Gradient)
{
	// SESSION sess = getSession("TENSOR::Gradient");

	// std::vector<ade::DimT> slist = get_shape(sess, "slist");
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf2 = ade::Tensor::get(ade::Shape(slist));

	ASSERT_NE(nullptr, leaf.get());

	ade::Tensorptr gotwun = leaf->gradient(leaf);
	ade::Tensorptr gotzro = leaf->gradient(leaf2);

	auto wunrp = dynamic_cast<ade::Functor<ade::RESHAPE,
		std::vector<ade::DimT>>*>(gotwun.get());
	auto zrorp = dynamic_cast<ade::Functor<ade::RESHAPE,
		std::vector<ade::DimT>>*>(gotzro.get());
	ASSERT_NE(nullptr, wunrp);
	ASSERT_NE(nullptr, zrorp);

	// std::string expect_label = sess->expect_string("expect_label");
	std::string expectlabel = opname(ade::RESHAPE) + "<[2\\3]>";
	EXPECT_STREQ(expectlabel.c_str(), wunrp->to_string().c_str());
	EXPECT_STREQ(expectlabel.c_str(), zrorp->to_string().c_str());
	// sess->store_string("expect_label", zrorp->to_string());

	EXPECT_ARREQ(slist, wunrp->shape_.as_list());
	EXPECT_ARREQ(slist, zrorp->shape_.as_list());

	std::vector<ade::iTensor*> wun_vec = wunrp->get_refs();
	std::vector<ade::iTensor*> zro_vec = zrorp->get_refs();
	ASSERT_EQ(1, wun_vec.size());
	ASSERT_EQ(1, zro_vec.size());
	EXPECT_EQ(ade::Tensor::SYMBOLIC_ONE.get(), wun_vec[0]);
	EXPECT_EQ(ade::Tensor::SYMBOLIC_ZERO.get(), zro_vec[0]);
}


TEST(TENSOR, ToString)
{
	// SESSION sess = getSession("TENSOR::ToString");

	// std::vector<ade::DimT> slist = get_shape(sess, "slist");
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));

	ASSERT_NE(nullptr, leaf.get());

	// std::string expect_out = sess->expect_string("expect_out");
	std::string expect_out = "[2\\3]";
	EXPECT_STREQ(expect_out.c_str(), leaf->to_string().c_str());
	// sess->store_string("expect_out", out);
}


#endif /* DISABLE_TENSOR_TEST */
