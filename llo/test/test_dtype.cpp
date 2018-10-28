
#ifndef DISABLE_DTYPE_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "llo/data.hpp"


struct DTYPE : public simple::TestModel
{
	virtual void TearDown (void)
	{
		simple::TestModel::TearDown();
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
	}
};


TEST_F(DTYPE, DataConvert)
{
	simple::SessionT sess = get_session("DTYPE::DataConvert");

	auto dtype = (llo::DTYPE) sess->get_scalar("dtype", {1, llo::_SENTINEL - 1});
	auto dtype_other = (llo::DTYPE) sess->get_scalar("dtype_other", {1, llo::_SENTINEL - 1});
	auto slist = get_shape(sess, "slist");
	ade::Shape shape(slist);
	llo::GenericData data(shape, dtype);
	std::memset(data.data_.get(), 0, type_size(dtype) * shape.n_elems()); // avoid memory issues
	llo::GenericData odata = data.convert_to(dtype_other);

	std::vector<ade::DimT> gotslist = odata.shape_.as_list();
	EXPECT_ARREQ(slist, gotslist);
	EXPECT_EQ(dtype_other, odata.dtype_);
}


TEST_F(DTYPE, GetType)
{
	EXPECT_EQ(llo::DOUBLE, llo::get_type<double>());
	EXPECT_EQ(llo::FLOAT, llo::get_type<float>());
	EXPECT_EQ(llo::INT8, llo::get_type<int8_t>());
	EXPECT_EQ(llo::UINT8, llo::get_type<uint8_t>());
	EXPECT_EQ(llo::INT16, llo::get_type<int16_t>());
	EXPECT_EQ(llo::UINT16, llo::get_type<uint16_t>());
	EXPECT_EQ(llo::INT32, llo::get_type<int32_t>());
	EXPECT_EQ(llo::UINT32, llo::get_type<uint32_t>());
	EXPECT_EQ(llo::INT64, llo::get_type<int64_t>());
	EXPECT_EQ(llo::UINT64, llo::get_type<uint64_t>());
	EXPECT_EQ(llo::BAD, llo::get_type<std::string>());
}


TEST_F(DTYPE, TypeSize)
{
	EXPECT_EQ(sizeof(double), llo::type_size(llo::DOUBLE));
	EXPECT_EQ(sizeof(float), llo::type_size(llo::FLOAT));
	EXPECT_EQ(sizeof(int8_t), llo::type_size(llo::INT8));
	EXPECT_EQ(sizeof(uint8_t), llo::type_size(llo::UINT8));
	EXPECT_EQ(sizeof(int16_t), llo::type_size(llo::INT16));
	EXPECT_EQ(sizeof(uint16_t), llo::type_size(llo::UINT16));
	EXPECT_EQ(sizeof(int32_t), llo::type_size(llo::INT32));
	EXPECT_EQ(sizeof(uint32_t), llo::type_size(llo::UINT32));
	EXPECT_EQ(sizeof(int64_t), llo::type_size(llo::INT64));
	EXPECT_EQ(sizeof(uint64_t), llo::type_size(llo::UINT64));
	EXPECT_FATAL(llo::type_size(llo::BAD), "unsupported type 0");
}


TEST_F(DTYPE, TypeName)
{
	EXPECT_STREQ("DOUBLE", llo::nametype(llo::DOUBLE).c_str());;
	EXPECT_STREQ("FLOAT", llo::nametype(llo::FLOAT).c_str());;
	EXPECT_STREQ("INT8", llo::nametype(llo::INT8).c_str());;
	EXPECT_STREQ("UINT8", llo::nametype(llo::UINT8).c_str());;
	EXPECT_STREQ("INT16", llo::nametype(llo::INT16).c_str());;
	EXPECT_STREQ("UINT16", llo::nametype(llo::UINT16).c_str());;
	EXPECT_STREQ("INT32", llo::nametype(llo::INT32).c_str());;
	EXPECT_STREQ("UINT32", llo::nametype(llo::UINT32).c_str());;
	EXPECT_STREQ("INT64", llo::nametype(llo::INT64).c_str());;
	EXPECT_STREQ("UINT64", llo::nametype(llo::UINT64).c_str());;
	EXPECT_STREQ("BAD_TYPE", llo::nametype(llo::BAD).c_str());;
}


#endif // DISABLE_DTYPE_TEST
