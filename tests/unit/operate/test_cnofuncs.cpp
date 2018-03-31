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

#include "operate/operations.hpp"


#ifndef DISABLE_CNOFUNCS_TEST // compound node functions


static const double ERR_THRESH = 0.08; // 8% error


struct CNOFUNCS : public testutils::fuzz_test {};





using namespace testutils;


#endif /* DISABLE_CNOFUNCS_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
