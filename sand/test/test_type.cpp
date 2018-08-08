#include "gtest/gtest.h"

#include "sand/type.hpp"


#ifndef DISABLE_TYPE_TEST


TEST(TYPE, TypeSize)
{
	EXPECT_EQ(sizeof(double), type_size(DTYPE::DOUBLE));
	EXPECT_EQ(sizeof(float), type_size(DTYPE::FLOAT));
	EXPECT_EQ(1, type_size(DTYPE::INT8));
	EXPECT_EQ(1, type_size(DTYPE::UINT8));
	EXPECT_EQ(2, type_size(DTYPE::INT16));
	EXPECT_EQ(2, type_size(DTYPE::UINT16));
	EXPECT_EQ(4, type_size(DTYPE::INT32));
	EXPECT_EQ(4, type_size(DTYPE::UINT32));
	EXPECT_EQ(8, type_size(DTYPE::INT64));
	EXPECT_EQ(8, type_size(DTYPE::UINT64));
	EXPECT_THROW(type_size(DTYPE::BAD), std::runtime_error);
}


TEST(TYPE, GetType)
{
	EXPECT_EQ(DTYPE::DOUBLE, get_type<double>());
	EXPECT_EQ(DTYPE::FLOAT, get_type<float>());
	EXPECT_EQ(DTYPE::INT8, get_type<int8_t>());
	EXPECT_EQ(DTYPE::UINT8, get_type<uint8_t>());
	EXPECT_EQ(DTYPE::INT16, get_type<int16_t>());
	EXPECT_EQ(DTYPE::UINT16, get_type<uint16_t>());
	EXPECT_EQ(DTYPE::INT32, get_type<int32_t>());
	EXPECT_EQ(DTYPE::UINT32, get_type<uint32_t>());
	EXPECT_EQ(DTYPE::INT64, get_type<int64_t>());
	EXPECT_EQ(DTYPE::UINT64, get_type<uint64_t>());
	EXPECT_EQ(DTYPE::BAD, get_type<std::string>());
}


TEST(TYPE, NameType)
{
	EXPECT_STREQ("DOUBLE", name_type(DOUBLE).c_str());
	EXPECT_STREQ("FLOAT", name_type(FLOAT).c_str());
	EXPECT_STREQ("INT8", name_type(INT8).c_str());
	EXPECT_STREQ("UINT8", name_type(UINT8).c_str());
	EXPECT_STREQ("INT16", name_type(INT16).c_str());
	EXPECT_STREQ("UINT16", name_type(UINT16).c_str());
	EXPECT_STREQ("INT32", name_type(INT32).c_str());
	EXPECT_STREQ("UINT32", name_type(UINT32).c_str());
	EXPECT_STREQ("INT64", name_type(INT64).c_str());
	EXPECT_STREQ("UINT64", name_type(UINT64).c_str());
	EXPECT_STREQ("BAD_TYPE", name_type(BAD).c_str());
}


#endif /* DISABLE_TYPE_TEST */
