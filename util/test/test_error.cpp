#include "gtest/gtest.h"

#include "util/error.hpp"


#ifndef DISABLE_ERROR_TEST


TEST(ERR, NoArg)
{
	std::string expect = "generic message";
	try
	{
		util::handle_error(expect);
		FAIL() << "handler failed to throw";
	}
	catch (std::runtime_error& e)
	{
		EXPECT_STREQ(expect.c_str(), e.what());
	}
	catch (...)
	{
		FAIL() << "handler failed to throw runtime_error";
	}
}


TEST(ERR, StringArg)
{
	std::string expect = "generic message[key\\"
		"\\\\abcd\\\\efgh\\\\ijkl\\\\]";
	try
	{
		util::handle_error("generic message",
			util::ErrArg<std::string>("key", "\\abcd\\efgh\\ijkl\\"));
		FAIL() << "handler failed to throw";
	}
	catch (std::runtime_error& e)
	{
		EXPECT_STREQ(expect.c_str(), e.what());
	}
	catch (...)
	{
		FAIL() << "handler failed to throw runtime_error";
	}
}


TEST(ERR, VectorArg)
{
	std::string expect = "generic message[key\\[-1\\21\\-13]]";
	try
	{
		util::handle_error("generic message",
			util::ErrArg<std::vector<int>>("key", {-1, 21, -13}));
		FAIL() << "handler failed to throw";
	}
	catch (std::runtime_error& e)
	{
		EXPECT_STREQ(expect.c_str(), e.what());
	}
	catch (...)
	{
		FAIL() << "handler failed to throw runtime_error";
	}
}


TEST(ERR, GenericArg)
{
	std::string expect = "generic message[key\\2.42333]";
	try
	{
		util::handle_error("generic message",
			util::ErrArg<float>("key", 2.42333));
		FAIL() << "handler failed to throw";
	}
	catch (std::runtime_error& e)
	{
		EXPECT_STREQ(expect.c_str(), e.what());
	}
	catch (...)
	{
		FAIL() << "handler failed to throw runtime_error";
	}
}


TEST(ERR, MultiArg)
{
	std::string expect = "generic message"
		"[str\\hey what's the answer to number]"
		"[int\\15][strvec\\[\\\\\\fine I'll tell you\\it's]]"
		"[dblvec\\[16.001\\-13.2\\45.2]]";
	try
	{
		util::handle_error("generic message",
			util::ErrArg<std::string>("str", "hey what's the answer to number"),
			util::ErrArg<int>("int", 15),
			util::ErrArg<std::vector<std::string>>("strvec", {
				"\\", "fine I'll tell you", "it's"}),
			util::ErrArg<std::vector<double>>("dblvec", {16.001, -13.2, 45.2}));
		FAIL() << "handler failed to throw";
	}
	catch (std::runtime_error& e)
	{
		EXPECT_STREQ(expect.c_str(), e.what());
	}
	catch (...)
	{
		FAIL() << "handler failed to throw runtime_error";
	}
}


#endif // DISABLE_ERROR_TEST
