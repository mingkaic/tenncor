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


TEST_F(CNOFUNCS, Argmax_B0xxAndB140)
{
}


TEST_F(CNOFUNCS, Rmax_B0xxAndB141)
{
}


TEST_F(CNOFUNCS, Rsum_B0xxAndB142)
{
}


TEST_F(CNOFUNCS, Transpose_B0xxAndB143)
{
}


TEST_F(CNOFUNCS, Flip_B0xxAndB144)
{
}


TEST_F(CNOFUNCS, ExpandB0xxAndB145)
{
}


TEST_F(CNOFUNCS, Nelems_B0xxAndB146)
{
}


TEST_F(CNOFUNCS, Ndims_B0xxAndB147)
{
}


TEST_F(CNOFUNCS, Clip_B0xxAndB148)
{
}


TEST_F(CNOFUNCS, ClipNorm_B0xxAndB149)
{
}


TEST_F(CNOFUNCS, Rmean_B0xxAndB150)
{
}


TEST_F(CNOFUNCS, Rnorm_B0xxAndB151)
{
}


using namespace testutils;


#endif /* DISABLE_CNOFUNCS_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
