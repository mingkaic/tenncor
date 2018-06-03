#ifndef DISABLE_CLAY_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/check.hpp"

#include "clay/dtype.hpp"
#include "clay/error.hpp"


#ifndef DISABLE_DTYPE_TEST


using namespace testutil;


class TYPE : public fuzz_test {};


TEST_F(TYPE, TypeSize_B000)
{
	EXPECT_EQ(sizeof(double), clay::type_size(clay::DTYPE::DOUBLE));
	EXPECT_EQ(sizeof(float), clay::type_size(clay::DTYPE::FLOAT));
	EXPECT_EQ(1, clay::type_size(clay::DTYPE::INT8));
	EXPECT_EQ(1, clay::type_size(clay::DTYPE::UINT8));
	EXPECT_EQ(2, clay::type_size(clay::DTYPE::INT16));
	EXPECT_EQ(2, clay::type_size(clay::DTYPE::UINT16));
	EXPECT_EQ(4, clay::type_size(clay::DTYPE::INT32));
	EXPECT_EQ(4, clay::type_size(clay::DTYPE::UINT32));
	EXPECT_EQ(8, clay::type_size(clay::DTYPE::INT64));
	EXPECT_EQ(8, clay::type_size(clay::DTYPE::UINT64));
	EXPECT_THROW(clay::type_size(clay::DTYPE::BAD), clay::UnsupportedTypeError);
}


TEST_F(TYPE, GetType_B001)
{
	EXPECT_EQ(clay::DTYPE::DOUBLE, clay::get_type<double>());
	EXPECT_EQ(clay::DTYPE::FLOAT, clay::get_type<float>());
	EXPECT_EQ(clay::DTYPE::INT8, clay::get_type<int8_t>());
	EXPECT_EQ(clay::DTYPE::UINT8, clay::get_type<uint8_t>());
	EXPECT_EQ(clay::DTYPE::INT16, clay::get_type<int16_t>());
	EXPECT_EQ(clay::DTYPE::UINT16, clay::get_type<uint16_t>());
	EXPECT_EQ(clay::DTYPE::INT32, clay::get_type<int32_t>());
	EXPECT_EQ(clay::DTYPE::UINT32, clay::get_type<uint32_t>());
	EXPECT_EQ(clay::DTYPE::INT64, clay::get_type<int64_t>());
	EXPECT_EQ(clay::DTYPE::UINT64, clay::get_type<uint64_t>());
	EXPECT_EQ(clay::DTYPE::BAD, clay::get_type<std::string>());
}


#endif /* DISABLE_DTYPE_TEST */


#endif /* DISABLE_CLAY_MODULE_TESTS */
