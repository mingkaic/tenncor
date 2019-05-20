#include "logs/logs.hpp"

#include "fmts/fmts.hpp"

#include "dbg/stream/ade.hpp"

#ifndef TESTUTIL_COMMON_HPP
#define TESTUTIL_COMMON_HPP

struct TestLogger : public logs::iLogger
{
	static std::string latest_warning_;
	static std::string latest_error_;

	enum LOG_LEVEL
	{
		NO = 0,
		YES,
	};

	void log (size_t log_level, std::string msg) const override
	{
		if (log_level <= log_level_)
		{
			switch (log_level_)
			{
				case NO:
					warn(msg);
					break;
				case YES:
				default:
					std::cout << msg << '\n';
			}
		}
	}

	size_t get_log_level (void) const override
	{
		return log_level_;
	}

	void set_log_level (size_t log_level) override
	{
		log_level_ = (LOG_LEVEL) log_level;
	}

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

	LOG_LEVEL log_level_;
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

#define EXPECT_HAS(SET, CONTENT) { auto et = SET.end();\
	EXPECT_TRUE(et != std::find(SET.begin(), et, CONTENT)) <<\
		"cannot find " << #CONTENT << " in " << #SET; }

#define EXPECT_FATAL(EVENT, MSG) try { EVENT; FAIL() << \
	"did not expect " << #EVENT << " to succeed"; } \
	catch (std::runtime_error& e) { EXPECT_STREQ(MSG, e.what()); }\
	catch (std::exception& e) { FAIL() << "unexpected throw " << e.what(); }

#define EXPECT_WARN(EVENT, MSG) EVENT;\
	EXPECT_STREQ(MSG, TestLogger::latest_warning_.c_str());

std::string compare_graph (std::istream& expectstr, ade::TensptrT root,
	bool showshape = true,
	LabelsMapT labels = {});

#endif // TESTUTIL_COMMON_HPP
