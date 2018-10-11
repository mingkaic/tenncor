
#include "gtest/gtest.h"

#include "testutil/common.hpp"

int main (int argc, char** argv)
{
	char* gen = getenv("GENERATE_MODE");
	simple::INIT(":32768", gen != nullptr);

	set_logger(std::static_pointer_cast<ade::iLogger>(tlogger));

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();

	simple::SHUTDOWN();
	return ret;
}