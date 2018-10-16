#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "llo/node.hpp"


#ifndef DISABLE_NODE_TEST


struct NODE : public simple::TestModel
{
	virtual void TearDown (void)
	{
		simple::TestModel::TearDown();
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
	}
};


TEST_F(NODE, MismatchSource)
{
	simple::SessionT sess = get_session("NODE::MismatchSource");

	auto slist = get_shape(sess, "slist");
	ade::Shape shape(slist);
	std::vector<double> data = sess->get_double("data", shape.n_elems() - 1);

	std::stringstream ss;
	ss << "data size " << data.size() <<
		" does not match shape " << shape.to_string();
	EXPECT_FATAL(llo::Source<double>::get(shape, data), ss.str().c_str());
}


TEST_F(NODE, SourceRetype)
{
	simple::SessionT sess = get_session("NODE::SourceRetype");

	auto slist = get_shape(sess, "slist");
	ade::Shape shape(slist);

	size_t n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n);
	llo::DataNode ptr = llo::Source<double>::get(shape, data);

	llo::GenericData gd = ptr.data(llo::UINT16);
	ASSERT_EQ(llo::UINT16, gd.dtype_);
	std::vector<ade::DimT> gotslist = gd.shape_.as_list();
	EXPECT_ARREQ(slist, gotslist);

	uint16_t* gotdata = (uint16_t*) gd.data_.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ((uint16_t) data[i], gotdata[i]);
	}
}


TEST_F(NODE, PlaceHolder)
{
	simple::SessionT sess = get_session("NODE::Placeholder");

	auto slist = get_shape(sess, "slist");
	ade::Shape shape(slist);
	size_t n = shape.n_elems();
	llo::PlaceHolder<double> pl(shape);

	llo::GenericData uninit_gd = pl.data(llo::DOUBLE);
	ASSERT_EQ(llo::DOUBLE, uninit_gd.dtype_);
	std::vector<ade::DimT> uninit_slist = uninit_gd.shape_.as_list();
	EXPECT_ARREQ(slist, uninit_slist);

	double* uninit_data = (double*) uninit_gd.data_.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(0, uninit_data[i]);
	}

	std::vector<double> data = sess->get_double("data", n);
	pl = data;
	llo::GenericData gd = pl.data(llo::DOUBLE);
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
