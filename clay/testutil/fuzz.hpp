#include <cstdlib>

#include "testify/fuzz/fuzz.hpp"

#ifndef TESTUTIL_FUZZ_HPP
#define TESTUTIL_FUZZ_HPP

namespace testutil
{

struct fuzz_test : public testify::fuzz_test
{
	fuzz_test (void)
	{
		print_enabled = nullptr == std::getenv("FUZZ_DISABLED");
	}
};

}

#endif /* TESTUTIL_FUZZ_HPP */
