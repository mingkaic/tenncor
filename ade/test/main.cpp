#include "gtest/gtest.h"

#include "simple/jack.hpp"

int main (int argc, char** argv)
{
	size_t nreps;
	char* nrepeats = getenv("GTEST_REPEAT");
	if (nrepeats == nullptr)
	{
		nreps = 100;
	}
	else
	{
		nreps = atoi(nrepeats);
	}
	char* gen = getenv("GENERATE_MODE");
	simple::INIT(":32768", gen != nullptr, nreps);

	::testing::InitGoogleTest(&argc, argv);
	int ret = RUN_ALL_TESTS();

	simple::SHUTDOWN();
	return ret;
}
