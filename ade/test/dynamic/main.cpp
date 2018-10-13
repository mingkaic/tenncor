
#include "gtest/gtest.h"

#include "testutil/common.hpp"

int main (int argc, char** argv)
{
	char* gen = getenv("GENERATE_MODE");
	simple::INIT("localhost:10000", gen != nullptr, "certs/server.crt");

	set_logger(std::static_pointer_cast<ade::iLogger>(tlogger));

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();

	simple::SHUTDOWN();
	return ret;
}
