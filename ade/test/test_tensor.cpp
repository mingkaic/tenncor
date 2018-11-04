
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

	EXPECT_EQ(ade::Tensor::SYMBOLIC_ONE.get(), gotwun.get());
	EXPECT_EQ(ade::Tensor::SYMBOLIC_ZERO.get(), gotzro.get());
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
