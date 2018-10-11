#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "llo/node.hpp"


#ifndef DISABLE_NODE_TEST


struct NODE : public TestModel {};


TEST_F(NODE, MismatchSource)
{
	SESSION sess = get_session("NODE::MismatchSource");

	auto slist = get_shape(sess, "slist");
	ade::Shape shape(slist);
	std::vector<double> data = sess->get_double("data", shape.n_elems() - 1);

	EXPECT_THROW(llo::Source<double>::get(shape, data), std::runtime_error);
}


TEST_F(NODE, SourceRetype)
{
	SESSION sess = get_session("NODE::SourceRetype");

	auto slist = get_shape(sess, "slist");
	ade::Shape shape(slist);

	size_t n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n);
	ade::Tensorptr ptr = llo::Source<double>::get(shape, data);
	llo::iEvaluable* evaler = dynamic_cast<llo::iEvaluable*>(ptr.get());
	ASSERT_NE(nullptr, evaler);

	llo::GenericData gd = evaler->evaluate(llo::UINT16);
	ASSERT_EQ(llo::UINT16, gd.dtype_);
	std::vector<ade::DimT> gotslist = gd.shape_.as_list();
	EXPECT_ARREQ(slist, gotslist);

	uint16_t* gotdata = (uint16_t*) gd.data_.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ((uint16_t) data[i], gotdata[i]);
	}
}


TEST_F(NODE, Placeholder)
{
	SESSION sess = get_session("NODE::Placeholder");

	auto slist = get_shape(sess, "slist");
	ade::Shape shape(slist);
	llo::Placeholder<double> pl(shape);

	size_t n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n);
	pl = data;
	llo::GenericData gd = llo::evaluate(llo::DOUBLE, pl.get());
	ASSERT_EQ(llo::DOUBLE, gd.dtype_);
	std::vector<ade::DimT> gotslist = gd.shape_.as_list();
	EXPECT_ARREQ(slist, gotslist);

	double* gotdata = (double*) gd.data_.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(data[i], gotdata[i]);
	}
}


#endif // DISABLE_NODE_TEST
