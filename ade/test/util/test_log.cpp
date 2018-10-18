#include "gtest/gtest.h"

#include "ade/test/util/common.hpp"

#ifndef DISABLE_LOG_TEST


struct LOG : public ::testing::Test
{
protected:
	void TearDown (void) override
	{
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
		TestLogger::latest_fatal_ = "";
	}
};


TEST_F(LOG, Warn)
{
	ade::warn("warning message");
	const char* cmsg = TestLogger::latest_warning_.c_str();
	EXPECT_STREQ("warning message", cmsg);
}


TEST_F(LOG, WarnFmt)
{
	ade::warnf("warning %.2f message %d with format %s", 4.15, 33, "applepie");
	const char* cmsg = TestLogger::latest_warning_.c_str();
	EXPECT_STREQ("warning 4.15 message 33 with format applepie", cmsg);
}


TEST_F(LOG, Error)
{
	ade::error("erroring message");
	const char* emsg = TestLogger::latest_error_.c_str();
	EXPECT_STREQ("erroring message", emsg);
}


TEST_F(LOG, ErrorFmt)
{
	ade::errorf("erroring %.3f message %d with format %s", 0.31, 7, "orange");
	const char* emsg = TestLogger::latest_error_.c_str();
	EXPECT_STREQ("erroring 0.310 message 7 with format orange", emsg);
}


TEST_F(LOG, Fatal)
{
	ade::fatal("fatal message");
	const char* fmsg = TestLogger::latest_fatal_.c_str();
	EXPECT_STREQ("fatal message", fmsg);
}


TEST_F(LOG, FatalFmt)
{
	ade::fatalf("fatal %.4f message %d with format %s", 3.1415967, -1, "plum");
	const char* fmsg = TestLogger::latest_fatal_.c_str();
	EXPECT_STREQ("fatal 3.1416 message -1 with format plum", fmsg);
}


#endif // DISABLE_LOG_TEST
