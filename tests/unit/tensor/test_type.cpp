//
// Created by Mingkai Chen on 2018-01-29.
//

#ifndef DISABLE_TENSOR_MODULE_TESTS

#include "gtest/gtest.h"

#include "tensor/type.hpp"
#include "utils/error.hpp"


#ifndef DISABLE_TYPE_TEST


TEST(TYPE, TypeSize_B000)
{
	EXPECT_EQ(sizeof(double), nnet::type_size(DOUBLE));
	EXPECT_EQ(sizeof(float), nnet::type_size(FLOAT));
	EXPECT_EQ(1, nnet::type_size(INT8));
	EXPECT_EQ(1, nnet::type_size(UINT8));
	EXPECT_EQ(2, nnet::type_size(INT16));
	EXPECT_EQ(2, nnet::type_size(UINT16));
	EXPECT_EQ(4, nnet::type_size(INT32));
	EXPECT_EQ(4, nnet::type_size(UINT32));
	EXPECT_EQ(8, nnet::type_size(INT64));
	EXPECT_EQ(8, nnet::type_size(UINT64));
	EXPECT_THROW(nnet::type_size(BAD_T), nnutils::unsupported_type_error);
}


TEST(TYPE, GetType_B001)
{
	EXPECT_EQ(DOUBLE, nnet::get_type<double>());
	EXPECT_EQ(FLOAT, nnet::get_type<float>());
	EXPECT_EQ(INT8, nnet::get_type<int8_t>());
	EXPECT_EQ(UINT8, nnet::get_type<uint8_t>());
	EXPECT_EQ(INT16, nnet::get_type<int16_t>());
	EXPECT_EQ(UINT16, nnet::get_type<uint16_t>());
	EXPECT_EQ(INT32, nnet::get_type<int32_t>());
	EXPECT_EQ(UINT32, nnet::get_type<uint32_t>());
	EXPECT_EQ(INT64, nnet::get_type<int64_t>());
	EXPECT_EQ(UINT64, nnet::get_type<uint64_t>());
	EXPECT_EQ(BAD_T, nnet::get_type<std::string>());
}


#endif /* DISABLE_TYPE_TEST */


#endif /* DISABLE_TENSOR_MODULE_TESTS */
