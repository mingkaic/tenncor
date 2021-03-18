#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/global/global.hpp"

int main (int argc, char** argv)
{
	global::set_logger(new exam::NoSupportLogger());

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
