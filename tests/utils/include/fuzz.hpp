#include <cstdlib>

#include "testify_cpp/include/fuzz/fuzz.hpp"

#ifndef TTEST_FUZZ_HPP
#define TTEST_FUZZ_HPP

namespace testutils
{

struct fuzz_test : public testify::fuzz_test
{
	fuzz_test (void)
	{
		print_enabled = nullptr == std::getenv("FUZZ_DISABLED");
	}
};

}

#endif /* TTEST_FUZZ_HPP */
