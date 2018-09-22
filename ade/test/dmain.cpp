#include "gtest/gtest.h"

#include "simple/jack.hpp"

int main (int argc, char** argv)
{
	char* gen = getenv("GENERATE_MODE");
	simple::INIT(":32768", gen != nullptr);

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();

	simple::SHUTDOWN();
	return ret;
}