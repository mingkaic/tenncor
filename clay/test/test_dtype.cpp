#ifndef DISABLE_CLAY_MODULE_TESTS

#include "gtest/gtest.h"

#include "testutil/fuzz.hpp"
#include "testutil/check.hpp"

#include "clay/dtype.hpp"


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
	EXPECT_THROW(clay::type_size(clay::DTYPE::BAD), std::exception);
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


TEST_F(TYPE, ConvertSV_B002)
{
	size_t dsize = get_int(1, "datadoub.size", {15, 271})[0];
	size_t usize = get_int(1, "datauint.size", {15, 271})[0];
	std::vector<double> ex_dbl = get_double(dsize, "datadoub", {0, 255});
	std::vector<size_t> temp = get_int(usize, "datauint", {0, 255});
	std::vector<uint16_t> ex_uit(temp.begin(), temp.end());
	
	std::string out;
	std::vector<double> dvec;
	std::vector<float> fvec;
	std::vector<int8_t> i8vec;
	std::vector<uint8_t> u8vec;
	std::vector<int16_t> i16vec;
	std::vector<uint16_t> u16vec;
	std::vector<int32_t> i32vec;
	std::vector<uint32_t> u32vec;
	std::vector<int64_t> i64vec;
	std::vector<uint64_t> u64vec;
	EXPECT_FALSE(clay::convert<double>(out, clay::DTYPE::BAD, dvec));
	EXPECT_FALSE(clay::convert<float>(out, clay::DTYPE::BAD, fvec));
	EXPECT_FALSE(clay::convert<int8_t>(out, clay::DTYPE::BAD, i8vec));
	EXPECT_FALSE(clay::convert<uint8_t>(out, clay::DTYPE::BAD, u8vec));
	EXPECT_FALSE(clay::convert<int16_t>(out, clay::DTYPE::BAD, i16vec));
	EXPECT_FALSE(clay::convert<uint16_t>(out, clay::DTYPE::BAD, u16vec));
	EXPECT_FALSE(clay::convert<int32_t>(out, clay::DTYPE::BAD, i32vec));
	EXPECT_FALSE(clay::convert<uint32_t>(out, clay::DTYPE::BAD, u32vec));
	EXPECT_FALSE(clay::convert<int64_t>(out, clay::DTYPE::BAD, i64vec));
	EXPECT_FALSE(clay::convert<uint64_t>(out, clay::DTYPE::BAD, u64vec));

	// convert with input of type double
	{
		clay::convert<double>(out, clay::DTYPE::DOUBLE, ex_dbl);
		const double* optr = (const double*) out.c_str();
		std::vector<double> outV(optr, optr + out.size() / sizeof(double));
		EXPECT_ARREQ(ex_dbl, outV);
	}
	
	{
		std::vector<float> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<double>(out, clay::DTYPE::FLOAT, ex_dbl);
		const float* optr = (const float*) out.c_str();
		std::vector<float> outV(optr, optr + out.size() / sizeof(float));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int8_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<double>(out, clay::DTYPE::INT8, ex_dbl);
		const int8_t* optr = (const int8_t*) out.c_str();
		std::vector<int8_t> outV(optr, optr + out.size() / sizeof(int8_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint8_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<double>(out, clay::DTYPE::UINT8, ex_dbl);
		const uint8_t* optr = (const uint8_t*) out.c_str();
		std::vector<uint8_t> outV(optr, optr + out.size() / sizeof(uint8_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int16_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<double>(out, clay::DTYPE::INT16, ex_dbl);
		const int16_t* optr = (const int16_t*) out.c_str();
		std::vector<int16_t> outV(optr, optr + out.size() / sizeof(int16_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint16_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<double>(out, clay::DTYPE::UINT16, ex_dbl);
		const uint16_t* optr = (const uint16_t*) out.c_str();
		std::vector<uint16_t> outV(optr, optr + out.size() / sizeof(uint16_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int32_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<double>(out, clay::DTYPE::INT32, ex_dbl);
		const int32_t* optr = (const int32_t*) out.c_str();
		std::vector<int32_t> outV(optr, optr + out.size() / sizeof(int32_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint32_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<double>(out, clay::DTYPE::UINT32, ex_dbl);
		const uint32_t* optr = (const uint32_t*) out.c_str();
		std::vector<uint32_t> outV(optr, optr + out.size() / sizeof(uint32_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int64_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<double>(out, clay::DTYPE::INT64, ex_dbl);
		const int64_t* optr = (const int64_t*) out.c_str();
		std::vector<int64_t> outV(optr, optr + out.size() / sizeof(int64_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint64_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<double>(out, clay::DTYPE::UINT64, ex_dbl);
		const uint64_t* optr = (const uint64_t*) out.c_str();
		std::vector<uint64_t> outV(optr, optr + out.size() / sizeof(uint64_t));
		EXPECT_ARREQ(expectV, outV);
	}

	// convert with input of type uint16
	{
		std::vector<double> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<uint16_t>(out, clay::DTYPE::DOUBLE, ex_uit);
		const double* optr = (const double*) out.c_str();
		std::vector<double> outV(optr, optr + out.size() / sizeof(double));
		EXPECT_ARREQ(ex_uit, outV);
	}
	
	{
		std::vector<float> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<uint16_t>(out, clay::DTYPE::FLOAT, ex_uit);
		const float* optr = (const float*) out.c_str();
		std::vector<float> outV(optr, optr + out.size() / sizeof(float));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int8_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<uint16_t>(out, clay::DTYPE::INT8, ex_uit);
		const int8_t* optr = (const int8_t*) out.c_str();
		std::vector<int8_t> outV(optr, optr + out.size() / sizeof(int8_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint8_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<uint16_t>(out, clay::DTYPE::UINT8, ex_uit);
		const uint8_t* optr = (const uint8_t*) out.c_str();
		std::vector<uint8_t> outV(optr, optr + out.size() / sizeof(uint8_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int16_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<uint16_t>(out, clay::DTYPE::INT16, ex_uit);
		const int16_t* optr = (const int16_t*) out.c_str();
		std::vector<int16_t> outV(optr, optr + out.size() / sizeof(int16_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		clay::convert<uint16_t>(out, clay::DTYPE::UINT16, ex_uit);
		const uint16_t* optr = (const uint16_t*) out.c_str();
		std::vector<uint16_t> outV(optr, optr + out.size() / sizeof(uint16_t));
		EXPECT_ARREQ(ex_uit, outV);
	}
	
	{
		std::vector<int32_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<uint16_t>(out, clay::DTYPE::INT32, ex_uit);
		const int32_t* optr = (const int32_t*) out.c_str();
		std::vector<int32_t> outV(optr, optr + out.size() / sizeof(int32_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint32_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<uint16_t>(out, clay::DTYPE::UINT32, ex_uit);
		const uint32_t* optr = (const uint32_t*) out.c_str();
		std::vector<uint32_t> outV(optr, optr + out.size() / sizeof(uint32_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int64_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<uint16_t>(out, clay::DTYPE::INT64, ex_uit);
		const int64_t* optr = (const int64_t*) out.c_str();
		std::vector<int64_t> outV(optr, optr + out.size() / sizeof(int64_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint64_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<uint16_t>(out, clay::DTYPE::UINT64, ex_uit);
		const uint64_t* optr = (const uint64_t*) out.c_str();
		std::vector<uint64_t> outV(optr, optr + out.size() / sizeof(uint64_t));
		EXPECT_ARREQ(expectV, outV);
	}
}


TEST_F(TYPE, ConvertVS_B003)
{
	size_t dsize = get_int(1, "datadoub.size", {15, 271})[0];
	size_t usize = get_int(1, "datauint.size", {15, 271})[0];
	std::vector<double> ex_dbl = get_double(dsize, "datadoub", {0, 255});
	std::vector<size_t> temp = get_int(usize, "datauint", {0, 255});
	std::vector<uint16_t> ex_uit(temp.begin(), temp.end());
	
	std::string data;
	std::vector<double> dvec;
	std::vector<float> fvec;
	std::vector<int8_t> i8vec;
	std::vector<uint8_t> u8vec;
	std::vector<int16_t> i16vec;
	std::vector<uint16_t> u16vec;
	std::vector<int32_t> i32vec;
	std::vector<uint32_t> u32vec;
	std::vector<int64_t> i64vec;
	std::vector<uint64_t> u64vec;
	EXPECT_FALSE(clay::convert<double>(dvec, data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert<float>(fvec, data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert<int8_t>(i8vec, data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert<uint8_t>(u8vec, data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert<int16_t>(i16vec, data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert<uint16_t>(u16vec, data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert<int32_t>(i32vec, data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert<uint32_t>(u32vec, data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert<int64_t>(i64vec, data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert<uint64_t>(u64vec, data, clay::DTYPE::BAD));

	// convert with input of type double
	double* dptr = &ex_dbl[0];
	data = std::string((char*) dptr, dsize * sizeof(double));
	{
		std::vector<double> outV;
		clay::convert<double>(outV, data, clay::DTYPE::DOUBLE);
		EXPECT_ARREQ(ex_dbl, outV);
	}
	
	{
		std::vector<float> outV;
		std::vector<float> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<float>(outV, data, clay::DTYPE::DOUBLE);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int8_t> outV;
		std::vector<int8_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<int8_t>(outV, data, clay::DTYPE::DOUBLE);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint8_t> outV;
		std::vector<uint8_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<uint8_t>(outV, data, clay::DTYPE::DOUBLE);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int16_t> outV;
		std::vector<int16_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<int16_t>(outV, data, clay::DTYPE::DOUBLE);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint16_t> outV;
		std::vector<uint16_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<uint16_t>(outV, data, clay::DTYPE::DOUBLE);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int32_t> outV;
		std::vector<int32_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<int32_t>(outV, data, clay::DTYPE::DOUBLE);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint32_t> outV;
		std::vector<uint32_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<uint32_t>(outV, data, clay::DTYPE::DOUBLE);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int64_t> outV;
		std::vector<int64_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<int64_t>(outV, data, clay::DTYPE::DOUBLE);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint64_t> outV;
		std::vector<uint64_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert<uint64_t>(outV, data, clay::DTYPE::DOUBLE);
		EXPECT_ARREQ(expectV, outV);
	}

	// convert with input of type uint16
	uint16_t* iptr = &ex_uit[0];
	data = std::string((char*) iptr, usize * sizeof(uint16_t));
	{
		std::vector<double> outV;
		std::vector<double> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<double>(outV, data, clay::DTYPE::UINT16);
		EXPECT_ARREQ(ex_uit, outV);
	}
	
	{
		std::vector<float> outV;
		std::vector<float> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<float>(outV, data, clay::DTYPE::UINT16);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int8_t> outV;
		std::vector<int8_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<int8_t>(outV, data, clay::DTYPE::UINT16);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint8_t> outV;
		std::vector<uint8_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<uint8_t>(outV, data, clay::DTYPE::UINT16);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int16_t> outV;
		std::vector<int16_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<int16_t>(outV, data, clay::DTYPE::UINT16);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint16_t> outV;
		clay::convert<uint16_t>(outV, data, clay::DTYPE::UINT16);
		EXPECT_ARREQ(ex_uit, outV);
	}
	
	{
		std::vector<int32_t> outV;
		std::vector<int32_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<int32_t>(outV, data, clay::DTYPE::UINT16);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint32_t> outV;
		std::vector<uint32_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<uint32_t>(outV, data, clay::DTYPE::UINT16);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int64_t> outV;
		std::vector<int64_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<int64_t>(outV, data, clay::DTYPE::UINT16);
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint64_t> outV;
		std::vector<uint64_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert<uint64_t>(outV, data, clay::DTYPE::UINT16);
		EXPECT_ARREQ(expectV, outV);
	}
}


TEST_F(TYPE, ConvertSS_B004)
{
	size_t dsize = get_int(1, "datadoub.size", {15, 271})[0];
	size_t usize = get_int(1, "datauint.size", {15, 271})[0];
	std::vector<double> ex_dbl = get_double(dsize, "datadoub", {0, 255});
	std::vector<size_t> temp = get_int(usize, "datauint", {0, 255});
	std::vector<uint16_t> ex_uit(temp.begin(), temp.end());
	
	std::string out;
	std::string data;
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::DOUBLE,
	data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::FLOAT,
	data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::INT8,
	data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::INT16,
	data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::INT32,
	data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::INT64,
	data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::UINT8,
	data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::UINT16,
	data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::UINT32,
	data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::UINT64,
	data, clay::DTYPE::BAD));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::BAD,
	data, clay::DTYPE::DOUBLE));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::BAD,
	data, clay::DTYPE::FLOAT));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::BAD,
	data, clay::DTYPE::INT8));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::BAD,
	data, clay::DTYPE::INT16));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::BAD,
	data, clay::DTYPE::INT32));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::BAD,
	data, clay::DTYPE::INT64));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::BAD,
	data, clay::DTYPE::UINT8));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::BAD,
	data, clay::DTYPE::UINT16));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::BAD,
	data, clay::DTYPE::UINT32));
	EXPECT_FALSE(clay::convert(out, clay::DTYPE::BAD,
	data, clay::DTYPE::UINT64));

	// convert with input of type double
	double* dptr = &ex_dbl[0];
	data = std::string((char*) dptr, dsize * sizeof(double));
	{
		clay::convert(out, clay::DTYPE::DOUBLE, data, clay::DTYPE::DOUBLE);
		const double* optr = (const double*) out.c_str();
		std::vector<double> outV(optr, optr + out.size() / sizeof(double));
		EXPECT_ARREQ(ex_dbl, outV);
	}
	
	{
		std::vector<float> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert(out, clay::DTYPE::FLOAT, data, clay::DTYPE::DOUBLE);
		const float* optr = (const float*) out.c_str();
		std::vector<float> outV(optr, optr + out.size() / sizeof(float));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int8_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert(out, clay::DTYPE::INT8, data, clay::DTYPE::DOUBLE);
		const int8_t* optr = (const int8_t*) out.c_str();
		std::vector<int8_t> outV(optr, optr + out.size() / sizeof(int8_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint8_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert(out, clay::DTYPE::UINT8, data, clay::DTYPE::DOUBLE);
		const uint8_t* optr = (const uint8_t*) out.c_str();
		std::vector<uint8_t> outV(optr, optr + out.size() / sizeof(uint8_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int16_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert(out, clay::DTYPE::INT16, data, clay::DTYPE::DOUBLE);
		const int16_t* optr = (const int16_t*) out.c_str();
		std::vector<int16_t> outV(optr, optr + out.size() / sizeof(int16_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint16_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert(out, clay::DTYPE::UINT16, data, clay::DTYPE::DOUBLE);
		const uint16_t* optr = (const uint16_t*) out.c_str();
		std::vector<uint16_t> outV(optr, optr + out.size() / sizeof(uint16_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int32_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert(out, clay::DTYPE::INT32, data, clay::DTYPE::DOUBLE);
		const int32_t* optr = (const int32_t*) out.c_str();
		std::vector<int32_t> outV(optr, optr + out.size() / sizeof(int32_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint32_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert(out, clay::DTYPE::UINT32, data, clay::DTYPE::DOUBLE);
		const uint32_t* optr = (const uint32_t*) out.c_str();
		std::vector<uint32_t> outV(optr, optr + out.size() / sizeof(uint32_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int64_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert(out, clay::DTYPE::INT64, data, clay::DTYPE::DOUBLE);
		const int64_t* optr = (const int64_t*) out.c_str();
		std::vector<int64_t> outV(optr, optr + out.size() / sizeof(int64_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint64_t> expectV(ex_dbl.begin(), ex_dbl.end());
		clay::convert(out, clay::DTYPE::UINT64, data, clay::DTYPE::DOUBLE);
		const uint64_t* optr = (const uint64_t*) out.c_str();
		std::vector<uint64_t> outV(optr, optr + out.size() / sizeof(uint64_t));
		EXPECT_ARREQ(expectV, outV);
	}

	// convert with input of type uint16
	uint16_t* iptr = &ex_uit[0];
	data = std::string((char*) iptr, usize * sizeof(uint16_t));
	{
		std::vector<double> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert(out, clay::DTYPE::DOUBLE, data, clay::DTYPE::UINT16);
		const double* optr = (const double*) out.c_str();
		std::vector<double> outV(optr, optr + out.size() / sizeof(double));
		EXPECT_ARREQ(ex_uit, outV);
	}
	
	{
		std::vector<float> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert(out, clay::DTYPE::FLOAT, data, clay::DTYPE::UINT16);
		const float* optr = (const float*) out.c_str();
		std::vector<float> outV(optr, optr + out.size() / sizeof(float));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int8_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert(out, clay::DTYPE::INT8, data, clay::DTYPE::UINT16);
		const int8_t* optr = (const int8_t*) out.c_str();
		std::vector<int8_t> outV(optr, optr + out.size() / sizeof(int8_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint8_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert(out, clay::DTYPE::UINT8, data, clay::DTYPE::UINT16);
		const uint8_t* optr = (const uint8_t*) out.c_str();
		std::vector<uint8_t> outV(optr, optr + out.size() / sizeof(uint8_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int16_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert(out, clay::DTYPE::INT16, data, clay::DTYPE::UINT16);
		const int16_t* optr = (const int16_t*) out.c_str();
		std::vector<int16_t> outV(optr, optr + out.size() / sizeof(int16_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		clay::convert(out, clay::DTYPE::UINT16, data, clay::DTYPE::UINT16);
		const uint16_t* optr = (const uint16_t*) out.c_str();
		std::vector<uint16_t> outV(optr, optr + out.size() / sizeof(uint16_t));
		EXPECT_ARREQ(ex_uit, outV);
	}
	
	{
		std::vector<int32_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert(out, clay::DTYPE::INT32, data, clay::DTYPE::UINT16);
		const int32_t* optr = (const int32_t*) out.c_str();
		std::vector<int32_t> outV(optr, optr + out.size() / sizeof(int32_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint32_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert(out, clay::DTYPE::UINT32, data, clay::DTYPE::UINT16);
		const uint32_t* optr = (const uint32_t*) out.c_str();
		std::vector<uint32_t> outV(optr, optr + out.size() / sizeof(uint32_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<int64_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert(out, clay::DTYPE::INT64, data, clay::DTYPE::UINT16);
		const int64_t* optr = (const int64_t*) out.c_str();
		std::vector<int64_t> outV(optr, optr + out.size() / sizeof(int64_t));
		EXPECT_ARREQ(expectV, outV);
	}
	
	{
		std::vector<uint64_t> expectV(ex_uit.begin(), ex_uit.end());
		clay::convert(out, clay::DTYPE::UINT64, data, clay::DTYPE::UINT16);
		const uint64_t* optr = (const uint64_t*) out.c_str();
		std::vector<uint64_t> outV(optr, optr + out.size() / sizeof(uint64_t));
		EXPECT_ARREQ(expectV, outV);
	}
}


#endif /* DISABLE_DTYPE_TEST */


#endif /* DISABLE_CLAY_MODULE_TESTS */
