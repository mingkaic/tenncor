#include "gtest/gtest.h"

#include "util/strify.hpp"


#ifndef DISABLE_STRIFY_TEST


TEST(STRIFY, StringFmt)
{
	std::stringstream ss;
	util::to_stream(ss, "abcdefghijkl");
	EXPECT_STREQ("abcdefghijkl", ss.str().c_str());
	ss.str("");

	util::to_stream(ss, "abcd\\efgh\\ijkl\\");
	EXPECT_STREQ("abcd\\\\efgh\\\\ijkl\\\\", ss.str().c_str());
	ss.str("");

	util::to_stream(ss, "\\abcd\\efgh\\ijkl");
	EXPECT_STREQ("\\\\abcd\\\\efgh\\\\ijkl", ss.str().c_str());
}


TEST(STRIFY, VectorFmt)
{
	std::stringstream ss;
	util::to_stream(ss, std::vector<int>{14, 15, 16});
	EXPECT_STREQ("[14\\15\\16]", ss.str().c_str());
	ss.str("");

	util::to_stream(ss, std::vector<std::string>{
		"what's\\up\\mybro", "nothing\\much\\fam", "\\hella\\lit"});
	EXPECT_STREQ("[what's\\\\up\\\\mybro\\nothing\\\\much\\\\fam\\\\\\hella\\\\lit]",
		ss.str().c_str());
	ss.str("");

	util::to_stream(ss, std::vector<int>{});
	EXPECT_STREQ("[]", ss.str().c_str());
}


TEST(STRIFY, GenericFmt)
{
	std::stringstream ss;
	util::to_stream(ss, -15);
	EXPECT_STREQ("-15", ss.str().c_str());
	ss.str("");

	util::to_stream(ss, 16.001);
	EXPECT_STREQ("16.001", ss.str().c_str());
}


TEST(STRIFY, MultiFmt)
{
	std::stringstream ss;
	util::to_stream(ss, "hey what's the answer to number", 15,
		std::vector<std::string>{"\\", "fine I'll tell you", "it's"},
		std::vector<double>{16.001, -13.2, 45.2});
	EXPECT_STREQ("hey what's the answer to number\\15\\[\\\\\\"
		"fine I'll tell you\\it's]\\[16.001\\-13.2\\45.2]", ss.str().c_str());
}


TEST(STRIFY, Tuple)
{
	std::tuple<std::string,double,std::vector<std::string>,std::vector<int>>
	tp{"hey what's the answer to number", 15.1,
		{"\\", "fine I'll tell you", "it's"}, {16, 13, 45}};
	std::string inorder = util::tuple_to_string(tp);
	EXPECT_STREQ("hey what's the answer to number\\15.1\\[\\\\\\"
		"fine I'll tell you\\it's]\\[16\\13\\45]", inorder.c_str());
	std::string outorder = util::tuple_to_string(tp,
		std::integer_sequence<size_t, 0, 3, 2, 1>());
	EXPECT_STREQ("hey what's the answer to number\\[16\\13\\45]\\[\\\\\\"
		"fine I'll tell you\\it's]\\15.1", outorder.c_str());
}


#endif /* DISABLE_STRIFY_TEST */
