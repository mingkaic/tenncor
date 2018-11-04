#include "gtest/gtest.h"

#include "log/test/common.hpp"

std::string TestLogger::latest_warning_;

std::string TestLogger::latest_error_;

std::string TestLogger::latest_fatal_;

std::shared_ptr<TestLogger> tlogger = std::make_shared<TestLogger>();

int main (int argc, char** argv)
{
	ade::set_logger(std::static_pointer_cast<ade::iLogger>(tlogger));
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
