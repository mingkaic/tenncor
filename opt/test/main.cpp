#include "gtest/gtest.h"

#include "teq/logs.hpp"

int main (int argc, char** argv)
{
	LOG_INIT(logs::DefLogger);

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
