#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "wire/variable.hpp"


#ifndef DISABLE_VARIABLE_TEST


using namespace testutil;


class VARIABLE : public fuzz_test {};


TEST_F(VARIABLE, Init_E000)
{
}


#endif /* DISABLE_VARIABLE_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */
