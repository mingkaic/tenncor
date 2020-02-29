#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/logs.hpp"

int main (int argc, char** argv)
{
	LOG_INIT(exam::TestLogger);

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
