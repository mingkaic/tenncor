
#ifndef DISABLE_LEAF_TEST


#include "gtest/gtest.h"

#include "internal/teq/ileaf.hpp"


TEST(LEAF, ConstEncoding)
{
	std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8};
	EXPECT_STREQ("1", teq::const_encode(
		data.data(), teq::Shape()).c_str());
	EXPECT_STREQ("[1\\2\\3\\4]", teq::const_encode(
		data.data(), teq::Shape({4})).c_str());
	EXPECT_STREQ("[1\\2\\3\\4]", teq::const_encode(
		data.data(), teq::Shape({4})).c_str());
	EXPECT_STREQ("[1\\2\\3\\4\\5\\...]", teq::const_encode(
		data.data(), teq::Shape({4,2})).c_str());
}


TEST(LEAF, GetUsage)
{
	auto immstr = teq::get_usage_name(teq::IMMUTABLE);
	auto varstr = teq::get_usage_name(teq::VARUSAGE);
	auto plcstr = teq::get_usage_name(teq::PLACEHOLDER);
	EXPECT_EQ(teq::IMMUTABLE, teq::get_named_usage(immstr));
	EXPECT_EQ(teq::VARUSAGE, teq::get_named_usage(varstr));
	EXPECT_EQ(teq::PLACEHOLDER, teq::get_named_usage(plcstr));
}


#endif // DISABLE_LEAF_TEST
