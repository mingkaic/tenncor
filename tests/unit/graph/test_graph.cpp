//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_OPERATE_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "fuzz.hpp"
#include "sgen.hpp"
#include "check.hpp"
#include "print.hpp"

#include "graph/graph.hpp"


#ifndef DISABLE_GRAPH_TEST // compound node functions


struct GRAPH : public testutils::fuzz_test {};


TEST_F(GRAPH, GraphSerialize_G000)
{
}


TEST_F(GRAPH, SerialConst_G001)
{
}


TEST_F(GRAPH, SerialPlace_G002)
{
}


TEST_F(GRAPH, SerialVar_G003)
{
}


TEST_F(GRAPH, SerialFunc_G004)
{
}


TEST_F(GRAPH, SerialData_G005)
{
}


#endif /* DISABLE_GRAPH_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
