
#ifndef TESTCASE_WITH_LOGGER_HPP
#define TESTCASE_WITH_LOGGER_HPP

#include "internal/global/global.hpp"

#include "exam/exam.hpp"

namespace tutil
{

template <typename DEFLOGGER=exam::NoSupportLogger, typename SWAPLOGGER=exam::MockLogger>
struct TestcaseWithLogger : public ::testing::Test
{
	virtual ~TestcaseWithLogger (void) = default;

	exam::MockLogger* logger_;

protected:
	void SetUp (void) override
	{
		logger_ = new SWAPLOGGER();
		global::set_logger(logger_);
	}

	void TearDown (void) override
	{
		global::set_logger(new DEFLOGGER());
		logger_ = nullptr;
	}
};

}

#endif // TESTCASE_WITH_LOGGER_HPP
