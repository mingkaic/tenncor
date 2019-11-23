
#ifndef DISABLE_LEAF_TEST


#include "gtest/gtest.h"

#include "teq/ileaf.hpp"


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


#endif // DISABLE_LEAF_TEST
