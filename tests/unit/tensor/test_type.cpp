//
// Created by Mingkai Chen on 2018-01-29.
//

#ifndef DISABLE_TENSOR_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzz.hpp"

#include "tensor/type.hpp"
#include "utils/error.hpp"


#ifndef DISABLE_TYPE_TEST


class TYPE : public testutils::fuzz_test {};


TEST_F(TYPE, TypeSize_B000)
{
	EXPECT_EQ(sizeof(double), nnet::type_size(nnet::DOUBLE));
	EXPECT_EQ(sizeof(float), nnet::type_size(nnet::FLOAT));
	EXPECT_EQ(1, nnet::type_size(nnet::INT8));
	EXPECT_EQ(1, nnet::type_size(nnet::UINT8));
	EXPECT_EQ(2, nnet::type_size(nnet::INT16));
	EXPECT_EQ(2, nnet::type_size(nnet::UINT16));
	EXPECT_EQ(4, nnet::type_size(nnet::INT32));
	EXPECT_EQ(4, nnet::type_size(nnet::UINT32));
	EXPECT_EQ(8, nnet::type_size(nnet::INT64));
	EXPECT_EQ(8, nnet::type_size(nnet::UINT64));
	EXPECT_THROW(nnet::type_size(nnet::BAD_T), nnutils::unsupported_type_error);
}


TEST_F(TYPE, GetType_B001)
{
	EXPECT_EQ(nnet::DOUBLE, nnet::get_type<double>());
	EXPECT_EQ(nnet::FLOAT, nnet::get_type<float>());
	EXPECT_EQ(nnet::INT8, nnet::get_type<int8_t>());
	EXPECT_EQ(nnet::UINT8, nnet::get_type<uint8_t>());
	EXPECT_EQ(nnet::INT16, nnet::get_type<int16_t>());
	EXPECT_EQ(nnet::UINT16, nnet::get_type<uint16_t>());
	EXPECT_EQ(nnet::INT32, nnet::get_type<int32_t>());
	EXPECT_EQ(nnet::UINT32, nnet::get_type<uint32_t>());
	EXPECT_EQ(nnet::INT64, nnet::get_type<int64_t>());
	EXPECT_EQ(nnet::UINT64, nnet::get_type<uint64_t>());
	EXPECT_EQ(nnet::BAD_T, nnet::get_type<std::string>());
}


TEST_F(TYPE, TypeConvert_B002)
{
	// size_t dsize = get_int(1, "datadoub.size", {15, 271})[0];
	// size_t isize = get_int(1, "dataint.size", {15, 271})[0];
	// std::vector<double> datad = get_double(dsize, "datadoub", {-271, 591});
	// auto temp = get_int(isize, "dataint", {0, 712});
	// std::vector<uint64_t> datai(temp.begin(), temp.end());

	size_t dsize = 206;
	size_t isize = 163;
	std::vector<double> datad = {384.604,564.175,577.793,175.448,-45.0862,-52.9049,528.14,282.459,344.76,369.437,387.776,161.766,-132.462,243.2,-34.4088,-149.29,-45.2492,321.871,45.3052,-46.1437,-38.935,-174.549,-251.373,512.788,-60.3895,-133.279,-180.971,65.1813,234.219,-91.7319,-88.6267,95.8528,176.904,70.9909,204.652,279.326,205.704,206.111,274.266,453.445,-164.134,-222.536,-221.944,-139.748,-15.5797,-204.007,527.284,135.105,237.951,-125.967,32.5188,8.22896,-167.745,-19.4958,128.189,511.925,163.99,-95.9459,458.97,385.939,267.986,-120.984,212.079,424.814,426.583,-50.8206,-173.436,397.141,241.704,-129.33,274.557,342.706,-19.303,-150.172,549.326,371.435,-163.786,-247.748,-7.45536,-93.9218,26.9053,76.6777,109.884,-246.443,423.575,63.7215,95.678,-71.3496,27.502,159.577,479.813,512.855,283.219,197.674,-253.709,-258.304,-61.2263,247.335,-160.783,-28.4829,330.758,500.509,528.414,237.404,-233.196,-147.5,135.129,168.038,-218.903,242.444,-199.016,-77.8961,116.843,436.54,-261.167,258.498,138.691,-9.48286,-242.734,164.475,250.577,70.8766,192.792,189.029,-245.454,242.106,147.003,92.7491,34.4031,433.052,211.594,72.123,-68.4823,199.014,-247.808,251.264,535.304,278.126,262.224,436.633,150.578,118.071,325.788,67.4419,180.379,-201.386,353.258,279.207,33.5508,-63.7022,42.3179,209.13,281.606,556.583,287.565,-88.9699,375.227,-129.411,280.527,-16.3534,46.7758,299.382,-176.091,350.182,450.549,26.5358,494.198,-36.7987,224.651,37.7124,303.708,385.792,-121.585,-264.003,35.0429,27.9169,38.1861,-138.159,366.868,529.719,-214.533,217.622,272.912,-60.5157,-245.699,504.58,-263.834,554.774,439.632,398.916,302.492,42.0844,14.3251,112.665,179.743,-108.518,231.54,8.70473,-144.805,14.1164,300.553,481.26,276.188,-80.6518,-16.1144,-251.993};
	std::vector<uint64_t> datai = {396,55,400,211,173,671,28,596,495,364,354,36,234,320,566,67,81,194,460,96,20,622,400,193,650,440,209,272,287,486,291,540,411,590,238,301,85,280,211,540,141,41,320,333,461,354,97,409,83,273,557,449,654,542,537,0,244,108,568,214,251,233,683,393,547,302,348,11,472,414,368,40,501,444,372,228,404,356,697,513,400,326,509,551,87,329,144,419,76,217,655,572,477,89,644,506,46,281,178,475,391,112,607,307,27,230,651,361,190,497,621,523,215,513,397,398,495,191,704,342,443,335,614,139,704,494,289,678,146,45,461,298,285,223,218,552,334,611,569,79,293,414,453,365,137,217,93,625,647,373,274,198,465,272,264,391,28,336,661,646,159,336,106};

	EXPECT_THROW(nnet::type_convert<double>(&datad[0], datad.size(), nnet::BAD_T), nnutils::unsupported_type_error);
	EXPECT_THROW(nnet::type_convert<double>(&datai[0], datai.size(), nnet::BAD_T), nnutils::unsupported_type_error);

	std::vector<double> dvec = nnet::type_convert<double>(&datad[0], datad.size(), nnet::DOUBLE);
	std::vector<float> fvec = nnet::type_convert<float>(&datad[0], datad.size(), nnet::DOUBLE);
	std::vector<int8_t> i8vec = nnet::type_convert<int8_t>(&datad[0], datad.size(), nnet::DOUBLE);
	std::vector<uint8_t> u8vec = nnet::type_convert<uint8_t>(&datad[0], datad.size(), nnet::DOUBLE);
	std::vector<int16_t> i16vec = nnet::type_convert<int16_t>(&datad[0], datad.size(), nnet::DOUBLE);
	std::vector<uint16_t> u16vec = nnet::type_convert<uint16_t>(&datad[0], datad.size(), nnet::DOUBLE);
	std::vector<int32_t> i32vec = nnet::type_convert<int32_t>(&datad[0], datad.size(), nnet::DOUBLE);
	std::vector<uint32_t> u32vec = nnet::type_convert<uint32_t>(&datad[0], datad.size(), nnet::DOUBLE);
	std::vector<int64_t> i64vec = nnet::type_convert<int64_t>(&datad[0], datad.size(), nnet::DOUBLE);
	std::vector<uint64_t> u64vec = nnet::type_convert<uint64_t>(&datad[0], datad.size(), nnet::DOUBLE);

	std::vector<double> dvec2 = nnet::type_convert<double>(&datai[0], datai.size(), nnet::UINT64);
	std::vector<float> fvec2 = nnet::type_convert<float>(&datai[0], datai.size(), nnet::UINT64);
	std::vector<int8_t> i8vec2 = nnet::type_convert<int8_t>(&datai[0], datai.size(), nnet::UINT64);
	std::vector<uint8_t> u8vec2 = nnet::type_convert<uint8_t>(&datai[0], datai.size(), nnet::UINT64);
	std::vector<int16_t> i16vec2 = nnet::type_convert<int16_t>(&datai[0], datai.size(), nnet::UINT64);
	std::vector<uint16_t> u16vec2 = nnet::type_convert<uint16_t>(&datai[0], datai.size(), nnet::UINT64);
	std::vector<int32_t> i32vec2 = nnet::type_convert<int32_t>(&datai[0], datai.size(), nnet::UINT64);
	std::vector<uint32_t> u32vec2 = nnet::type_convert<uint32_t>(&datai[0], datai.size(), nnet::UINT64);
	std::vector<int64_t> i64vec2 = nnet::type_convert<int64_t>(&datai[0], datai.size(), nnet::UINT64);
	std::vector<uint64_t> u64vec2 = nnet::type_convert<uint64_t>(&datai[0], datai.size(), nnet::UINT64);

	ASSERT_EQ(dsize, dvec.size());
	ASSERT_EQ(dsize, fvec.size());
	ASSERT_EQ(dsize, i8vec.size());
	ASSERT_EQ(dsize, u8vec.size());
	ASSERT_EQ(dsize, i16vec.size());
	ASSERT_EQ(dsize, u16vec.size());
	ASSERT_EQ(dsize, i32vec.size());
	ASSERT_EQ(dsize, u32vec.size());
	ASSERT_EQ(dsize, i64vec.size());
	ASSERT_EQ(dsize, u64vec.size());
	for (size_t i = 0; i < dsize; ++i)
	{
		EXPECT_EQ((double) datad[i], dvec[i]);
		EXPECT_EQ((float) datad[i], fvec[i]);
		EXPECT_EQ((int8_t) datad[i], i8vec[i]);
		EXPECT_EQ((uint8_t) datad[i], u8vec[i]);
		EXPECT_EQ((int16_t) datad[i], i16vec[i]);
		EXPECT_EQ((uint16_t) datad[i], u16vec[i]);
		EXPECT_EQ((int32_t) datad[i], i32vec[i]);
		EXPECT_EQ((uint32_t) datad[i], u32vec[i]);
		EXPECT_EQ((int64_t) datad[i], i64vec[i]);
		EXPECT_EQ((uint64_t) datad[i], u64vec[i]);
	}

	ASSERT_EQ(isize, dvec2.size());
	ASSERT_EQ(isize, fvec2.size());
	ASSERT_EQ(isize, i8vec2.size());
	ASSERT_EQ(isize, u8vec2.size());
	ASSERT_EQ(isize, i16vec2.size());
	ASSERT_EQ(isize, u16vec2.size());
	ASSERT_EQ(isize, i32vec2.size());
	ASSERT_EQ(isize, u32vec2.size());
	ASSERT_EQ(isize, i64vec2.size());
	ASSERT_EQ(isize, u64vec2.size());
	for (size_t i = 0; i < isize; ++i)
	{
		EXPECT_EQ((double) datai[i], dvec2[i]);
		EXPECT_EQ((float) datai[i], fvec2[i]);
		EXPECT_EQ((int8_t) datai[i], i8vec2[i]);
		EXPECT_EQ((uint8_t) datai[i], u8vec2[i]);
		EXPECT_EQ((int16_t) datai[i], i16vec2[i]);
		EXPECT_EQ((uint16_t) datai[i], u16vec2[i]);
		EXPECT_EQ((int32_t) datai[i], i32vec2[i]);
		EXPECT_EQ((uint32_t) datai[i], u32vec2[i]);
		EXPECT_EQ((int64_t) datai[i], i64vec2[i]);
		EXPECT_EQ((uint64_t) datai[i], u64vec2[i]);
	}
}


#endif /* DISABLE_TYPE_TEST */


#endif /* DISABLE_TENSOR_MODULE_TESTS */
