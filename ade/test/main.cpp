
#include "gtest/gtest.h"

#include "testutil/common.hpp"

int main (int argc, char** argv)
{
	set_logger(std::static_pointer_cast<err::iLogger>(tlogger));

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
