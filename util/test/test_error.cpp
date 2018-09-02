#include "gtest/gtest.h"

#include "util/error.hpp"


#ifndef DISABLE_ERROR_TEST


TEST(ERROR, NoArg)
{
	std::string expect = "generic message";
	try
	{
		handle_error(expect);
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


TEST(ERROR, StringArg)
{
	std::string expect = "generic message[key\\"
		"\\\\abcd\\\\efgh\\\\ijkl\\\\]";
	try
	{
		handle_error("generic message",
			ErrArg<std::string>("key", "\\abcd\\efgh\\ijkl\\"));
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


TEST(ERROR, VectorArg)
{
	std::string expect = "generic message[key\\[-1\\21\\-13]]";
	try
	{
		handle_error("generic message",
			ErrArg<std::vector<int>>("key", {-1, 21, -13}));
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


TEST(ERROR, GenericArg)
{
	std::string expect = "generic message[key\\2.42333]";
	try
	{
		handle_error("generic message",
			ErrArg<float>("key", 2.42333));
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


TEST(ERROR, MultiArg)
{
	std::string expect = "generic message"
		"[str\\hey what's the answer to number]"
		"[int\\15][strvec\\[\\\\\\fine I'll tell you\\it's]]"
		"[dblvec\\[16.001\\-13.2\\45.2]]";
	try
	{
		handle_error("generic message",
			ErrArg<std::string>("str", "hey what's the answer to number"),
			ErrArg<int>("int", 15),
			ErrArg<std::vector<std::string>>("strvec", {
				"\\", "fine I'll tell you", "it's"}),
			ErrArg<std::vector<double>>("dblvec", {16.001, -13.2, 45.2}));
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


#endif /* DISABLE_ERROR_TEST */
