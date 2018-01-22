//
// Created by Mingkai Chen on 2018-01-21.
//

#ifndef DISABLE_CONNECTOR_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

// #include "tests/include/mocks/mock_shape_dep.h"
// #include "tests/include/mocks/mock_node.h"
// #include "tests/include/mocks/mock_itensor.h"

// #include "include/graph/leaf/variable.hpp"
#include "tests/include/utils/fuzz.h"


#ifndef DISABLE_SHAPE_DEP_TEST


class SHAPE_DEP : public FUZZ::fuzz_test {};


TEST_F(SHAPE_DEP, DISABLED_Copy_)
{
	// todo: implement + add to behavior.txt
}


TEST_F(SHAPE_DEP, DISABLED_Move_)
{
	// todo: implement + add to behavior.txt
}


#endif /* DISABLE_SHAPE_DEP_TEST */


#endif /* DISABLE_CONNECTOR_MODULE_TESTS */
