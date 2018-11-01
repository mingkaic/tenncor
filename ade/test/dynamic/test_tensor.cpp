
#ifndef DISABLE_TENSOR_TEST


#include "gtest/gtest.h"

#include "ade/tensor.hpp"
#include "ade/functor.hpp"

#include "testutil/common.hpp"


struct TENSOR : public simple::TestModel {};


TEST_F(TENSOR, Gradient)
{
	simple::SessionT sess = get_session("TENSOR::Gradient");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf2 = ade::Tensor::get(ade::Shape(slist));

	ASSERT_NE(nullptr, leaf.get());

	ade::Tensorptr gotwun = leaf->gradient(leaf.get());
	ade::Tensorptr gotzro = leaf->gradient(leaf2.get());

	auto zrorp = dynamic_cast<ade::Functor*>(gotzro.get());
	auto wunrp = dynamic_cast<ade::Functor*>(gotwun.get());
	ASSERT_NE(nullptr, zrorp);
	ASSERT_NE(nullptr, wunrp);

	optional<std::string> expect_label = sess->expect_string("expect_label");
	if (expect_label)
	{
		EXPECT_STREQ(expect_label->c_str(), zrorp->to_string().c_str());
		EXPECT_STREQ(expect_label->c_str(), wunrp->to_string().c_str());
	}
	sess->store_string("expect_label", wunrp->to_string());

	// check zero
	std::vector<ade::DimT> zlist(zrorp->shape().begin(), zrorp->shape().end());
	EXPECT_ARREQ(slist, zlist);

	ade::ArgsT zro_vec = zrorp->get_children();
	ASSERT_EQ(1, zro_vec.size());
	EXPECT_EQ(ade::Tensor::SYMBOLIC_ZERO.get(), zro_vec[0].tensor_.get());

	// check one
	std::vector<ade::DimT> wlist(wunrp->shape().begin(), wunrp->shape().end());
	EXPECT_ARREQ(slist, wlist);

	ade::ArgsT wun_vec = wunrp->get_children();
	ASSERT_EQ(1, wun_vec.size());
	EXPECT_EQ(ade::Tensor::SYMBOLIC_ONE.get(), wun_vec[0].tensor_.get());
}


TEST_F(TENSOR, ToString)
{
	simple::SessionT sess = get_session("TENSOR::ToString");

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


#endif // DISABLE_TENSOR_TEST
