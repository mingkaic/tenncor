
#include "gtest/gtest.h"

#include "internal/global/global.hpp"

int main (int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	return ret;
}
