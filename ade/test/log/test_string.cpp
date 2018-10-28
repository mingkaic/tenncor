
#ifndef DISABLE_STRING_TEST


#include <array>

#include "gtest/gtest.h"

#include "ade/log/string.hpp"


TEST(STRING, StringFmt)
{
	std::stringstream ss;
	ade::to_stream(ss, "abcdefghijkl");
	EXPECT_STREQ("abcdefghijkl", ss.str().c_str());
	ss.str("");

	ade::to_stream(ss, "abcd\\efgh\\ijkl\\");
	EXPECT_STREQ("abcd\\\\efgh\\\\ijkl\\\\", ss.str().c_str());
	ss.str("");

	ade::to_stream(ss, "\\abcd\\efgh\\ijkl");
	EXPECT_STREQ("\\\\abcd\\\\efgh\\\\ijkl", ss.str().c_str());
}


TEST(STRING, VectorFmt)
{
	std::stringstream ss;
	ade::to_stream(ss, std::vector<int>{14, 15, 16});
	EXPECT_STREQ("[14\\15\\16]", ss.str().c_str());
	ss.str("");

	ade::to_stream(ss, std::vector<std::string>{
		"what's\\up\\mybro", "nothing\\much\\fam", "\\hella\\lit"});
	EXPECT_STREQ("[what's\\\\up\\\\mybro\\nothing\\\\much\\\\fam\\\\\\hella\\\\lit]",
		ss.str().c_str());
	ss.str("");

	ade::to_stream(ss, std::vector<int>{});
	EXPECT_STREQ("[]", ss.str().c_str());
}


TEST(STRING, GenericFmt)
{
	std::stringstream ss;
	ade::to_stream(ss, -15);
	EXPECT_STREQ("-15", ss.str().c_str());
	ss.str("");

	ade::to_stream(ss, 16.001);
	EXPECT_STREQ("16.001", ss.str().c_str());
}


TEST(STRING, MultiFmt)
{
	std::stringstream ss;
	ade::to_stream(ss, "hey what's the answer to number", 15,
		std::vector<std::string>{"\\", "fine I'll tell you", "it's"},
		std::vector<double>{16.001, -13.2, 45.2});
	EXPECT_STREQ("hey what's the answer to number\\15\\[\\\\\\"
		"fine I'll tell you\\it's]\\[16.001\\-13.2\\45.2]", ss.str().c_str());
}


TEST(STRING, Iterators)
{
	std::vector<double> dbs = {1.5, 1, 5.6, 7.8};
	std::array<int,4> iar = {-5, 2, -3, 6};
	std::string dbstr = ade::to_string(dbs.begin(), dbs.end());
	std::string iarstr = ade::to_string(iar.begin(), iar.end());
	EXPECT_STREQ("[1.5\\1\\5.6\\7.8]", dbstr.c_str());
	EXPECT_STREQ("[-5\\2\\-3\\6]", iarstr.c_str());
}


TEST(STRING, Sprintf)
{
	const char* str = "string";
	const char* oth = "other";
	std::string s = ade::sprintf("%% %s %d %.3f %% %%%% %s %d %.1f %% %%%%",
		str, 123, 5.689, oth, 77, 0.4, 12);
	EXPECT_STREQ("% string 123 5.689 % %% other 77 0.4 % %%", s.c_str());
}


#endif // DISABLE_STRING_TEST
