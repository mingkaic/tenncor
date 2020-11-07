
#ifndef DISABLE_EIGEN_META_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/eigen/eigen.hpp"


TEST(META, Types)
{
	eigen::EMetadata<double> dmeta;
	eigen::EMetadata<float> fmeta;
	eigen::EMetadata<int32_t> imeta;

	EXPECT_EQ(egen::DOUBLE, dmeta.type_code());
	EXPECT_EQ(egen::FLOAT, fmeta.type_code());
	EXPECT_EQ(egen::INT32, imeta.type_code());

	EXPECT_STREQ(egen::name_type(egen::DOUBLE).c_str(),
		dmeta.type_label().c_str());
	EXPECT_STREQ(egen::name_type(egen::FLOAT).c_str(),
		fmeta.type_label().c_str());
	EXPECT_STREQ(egen::name_type(egen::INT32).c_str(),
		imeta.type_label().c_str());

	EXPECT_EQ(sizeof(double), dmeta.type_size());
	EXPECT_EQ(sizeof(float), fmeta.type_size());
	EXPECT_EQ(sizeof(int32_t), imeta.type_size());
}


TEST(META, Version)
{
	eigen::EMetadata<double> dmeta;
	EXPECT_EQ(0, dmeta.state_version());
}


#endif // DISABLE_EIGEN_META_TEST
