#include "logs/logs.hpp"
#include "fmts/fmts.hpp"

struct TestLogger : public logs::iLogger
{
	static std::string latest_warning_;
	static std::string latest_error_;

	void warn (std::string msg) const override
	{
		latest_warning_ = msg;
	}

	void error (std::string msg) const override
	{
		latest_error_ = msg;
	}

	void fatal (std::string msg) const override
	{
		throw std::runtime_error(msg);
	}
};

extern std::shared_ptr<TestLogger> tlogger;

const size_t nelem_limit = 32456;

#define ASSERT_ARREQ(ARR, ARR2) { std::stringstream arrs, arrs2;\
	fmts::to_stream(arrs, ARR.begin(), ARR.end());\
	fmts::to_stream(arrs2, ARR2.begin(), ARR2.end());\
	ASSERT_TRUE(std::equal(ARR.begin(), ARR.end(), ARR2.begin())) <<\
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead"; }

#define EXPECT_ARREQ(ARR, ARR2) { std::stringstream arrs, arrs2;\
	fmts::to_stream(arrs, ARR.begin(), ARR.end());\
	fmts::to_stream(arrs2, ARR2.begin(), ARR2.end());\
	EXPECT_TRUE(std::equal(ARR.begin(), ARR.end(), ARR2.begin())) <<\
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead"; }

#define EXPECT_FATAL(EVENT, MSG) try { EVENT; FAIL() << \
	"did not expect " << #EVENT << " to succeed"; } \
	catch (std::runtime_error& e) { EXPECT_STREQ(MSG, e.what()); }\
	catch (std::exception& e) { FAIL() << "unexpected throw " << e.what(); }

#define EXPECT_WARN(EVENT, MSG) EVENT;\
	EXPECT_STREQ(MSG, TestLogger::latest_warning_.c_str());
