#include "gtest/gtest.h"

#include "ade/tensor.hpp"
#include "ade/functor.hpp"

#include "ade/test/common.hpp"


#ifndef DISABLE_TENSOR_TEST


struct TENSOR : public TestModel {};


TEST_F(TENSOR, Gradient)
{
	SESSION sess = get_session("TENSOR::Gradient");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
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

	optional<std::string> expect_label = sess->expect_string("expect_label");
	if (expect_label)
	{
		EXPECT_STREQ(expect_label->c_str(), wunrp->to_string().c_str());
		EXPECT_STREQ(expect_label->c_str(), zrorp->to_string().c_str());
	}
	EXPECT_STREQ(zrorp->to_string().c_str(), wunrp->to_string().c_str());
	sess->store_string("expect_label", zrorp->to_string());

	EXPECT_ARREQ(slist, wunrp->shape().as_list());
	EXPECT_ARREQ(slist, zrorp->shape().as_list());

	std::vector<ade::iTensor*> wun_vec = wunrp->get_refs();
	std::vector<ade::iTensor*> zro_vec = zrorp->get_refs();
	ASSERT_EQ(1, wun_vec.size());
	ASSERT_EQ(1, zro_vec.size());
	EXPECT_EQ(ade::Tensor::SYMBOLIC_ONE.get(), wun_vec[0]);
	EXPECT_EQ(ade::Tensor::SYMBOLIC_ZERO.get(), zro_vec[0]);
}


TEST_F(TENSOR, ToString)
{
	SESSION sess = get_session("TENSOR::ToString");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));

	ASSERT_NE(nullptr, leaf.get());

	optional<std::string> expect_out = sess->expect_string("expect_out");
	if (expect_out)
	{
		EXPECT_STREQ(expect_out->c_str(), leaf->to_string().c_str());
	}
	sess->store_string("expect_out", leaf->to_string());
}


#endif /* DISABLE_TENSOR_TEST */
