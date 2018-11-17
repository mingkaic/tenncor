
#ifndef DISABLE_LOG_TEST


#include "gtest/gtest.h"

#include "err/test/common.hpp"


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


TEST_F(LOG, Default)
{
	err::DefLogger log;
	log.warn("warning message");
	log.error("error message");
	try
	{
		log.fatal("fatal message");
		FAIL() << "log.fatal failed to throw error";
	}
	catch (std::runtime_error& e)
	{
		const char* msg = e.what();
		EXPECT_STREQ("fatal message", msg);
	}
	catch (...)
	{
		FAIL() << "expected to throw runtime_error";
	}
}


TEST_F(LOG, Warn)
{
	err::warn("warning message");
	const char* cmsg = TestLogger::latest_warning_.c_str();
	EXPECT_STREQ("warning message", cmsg);
}


TEST_F(LOG, WarnFmt)
{
	err::warnf("warning %.2f message %d with format %s", 4.15, 33, "applepie");
	const char* cmsg = TestLogger::latest_warning_.c_str();
	EXPECT_STREQ("warning 4.15 message 33 with format applepie", cmsg);
}


TEST_F(LOG, Error)
{
	err::error("erroring message");
	const char* emsg = TestLogger::latest_error_.c_str();
	EXPECT_STREQ("erroring message", emsg);
}


TEST_F(LOG, ErrorFmt)
{
	err::errorf("erroring %.3f message %d with format %s", 0.31, 7, "orange");
	const char* emsg = TestLogger::latest_error_.c_str();
	EXPECT_STREQ("erroring 0.310 message 7 with format orange", emsg);
}


TEST_F(LOG, Fatal)
{
	err::fatal("fatal message");
	const char* fmsg = TestLogger::latest_fatal_.c_str();
	EXPECT_STREQ("fatal message", fmsg);
}


TEST_F(LOG, FatalFmt)
{
	err::fatalf("fatal %.4f message %d with format %s", 3.1415967, -1, "plum");
	const char* fmsg = TestLogger::latest_fatal_.c_str();
	EXPECT_STREQ("fatal 3.1416 message -1 with format plum", fmsg);
}


#endif // DISABLE_LOG_TEST
