
#ifndef DISABLE_OPT_APPLY_TEST

#include <sstream>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/opt/mock/mock.hpp"


TEST(APPLY, DISABLED_Optimize)
{
	opt::GraphInfo graph({});
	opt::optimize(graph, {});
}


#endif // DISABLE_OPT_APPLY_TEST
