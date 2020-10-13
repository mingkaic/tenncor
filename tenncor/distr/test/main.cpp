
#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/global/global.hpp"

int main (int argc, char** argv)
{
	// global::set_logger(new exam::TestLogger());

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	return ret;
}
