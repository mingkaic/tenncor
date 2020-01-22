#include "gtest/gtest.h"

#include "teq/logs.hpp"

#include "eigen/device.hpp"
#include "eigen/random.hpp"

int main (int argc, char** argv)
{
	LOG_INIT(logs::DefLogger);
	DEVICE_INIT(eigen::Device);
	RANDOM_INIT;

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
