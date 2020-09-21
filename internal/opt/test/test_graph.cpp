
#ifndef DISABLE_OPT_GRAPH_TEST

#include <sstream>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/opt/mock/mock.hpp"


TEST(GRAPH, DISABLED_GetInfo)
{
	opt::GraphInfo graph({});
	graph.get_roots();
	graph.get_owners();
	graph.get_owner(nullptr);
}


TEST(Graph, DISABLED_Find)
{
	opt::GraphInfo graph({});
}


TEST(Graph, DISABLED_Replace)
{
	opt::GraphInfo graph({});
}


#endif // DISABLE_OPT_GRAPH_TEST
