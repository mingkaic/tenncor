#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "wire/placeholder.hpp"


#ifndef DISABLE_PLACEHOLDER_TEST


using namespace testutil;


class PLACEHOLDER : public fuzz_test {};


TEST_F(PLACEHOLDER, Assign_F000)
{
}


#endif /* DISABLE_PLACEHOLDER_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */
