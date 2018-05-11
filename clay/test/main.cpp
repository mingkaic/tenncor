#include <ctime>
#include <random>

#include "gtest/gtest.h"

#include "testify/fuzz/irng.hpp"

int main (int argc, char** argv)
{
	size_t seed = std::time(nullptr);
	std::default_random_engine gen(seed);
	testify::set_generator([&gen]() -> std::default_random_engine&
	{
		return gen;
	});
	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();
	return ret;
}
