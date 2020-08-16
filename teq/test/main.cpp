
#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "global/global.hpp"

int main (int argc, char** argv)
{
	global::set_logger(new exam::TestLogger());

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
