#include "gtest/gtest.h"

#include "err/test/common.hpp"

std::string TestLogger::latest_warning_;

std::string TestLogger::latest_error_;

std::string TestLogger::latest_fatal_;

std::shared_ptr<TestLogger> tlogger = std::make_shared<TestLogger>();

int main (int argc, char** argv)
{
	err::set_logger(std::static_pointer_cast<err::iLogger>(tlogger));
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
