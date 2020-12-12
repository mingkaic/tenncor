
#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/global/global.hpp"

int main (int argc, char** argv)
{
	global::set_logger(new exam::NoSupportLogger());

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	return ret;
}
