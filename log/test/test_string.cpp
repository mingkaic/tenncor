
#ifndef DISABLE_STRING_TEST


#include <array>
#include <list>
#include <vector>

#include "gtest/gtest.h"

#include "log/string.hpp"


TEST(STREAM, StringFmt)
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


TEST(STREAM, GenericFmt)
{
	std::stringstream ss;
	ade::to_stream(ss, -15);
	EXPECT_STREQ("-15", ss.str().c_str());
	ss.str("");

	ade::to_stream(ss, 16.001);
	EXPECT_STREQ("16.001", ss.str().c_str());
}


TEST(STREAM, Iterators)
{
	std::stringstream ss;
	std::vector<int> ivec = {14, 15, 16};
	ade::to_stream(ss, ivec.begin(), ivec.end());
	EXPECT_STREQ("[14\\15\\16]", ss.str().c_str());
	ss.str("");

	std::vector<std::string> svec = {
		"what's\\up\\mybro", "nothing\\much\\fam", "\\hella\\lit"};
	ade::to_stream(ss, svec.begin(), svec.end());
	EXPECT_STREQ("[what's\\\\up\\\\mybro\\nothing\\\\much\\\\fam\\\\\\hella\\\\lit]",
		ss.str().c_str());
	ss.str("");

	std::vector<int> emptyvec;
	ade::to_stream(ss, emptyvec.begin(), emptyvec.end());
	EXPECT_STREQ("[]", ss.str().c_str());
	ss.str("");

	std::array<uint8_t,4> uarr = {9, 0, 3, 36};
	std::list<int8_t> ilist = {-5, -2, 13, 61};
	ade::to_stream(ss, uarr.begin(), uarr.end());
	EXPECT_STREQ("[9\\0\\3\\36]", ss.str().c_str());
	ss.str("");

	ade::to_stream(ss, ilist.begin(), ilist.end());
	EXPECT_STREQ("[-5\\-2\\13\\61]", ss.str().c_str());
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
