//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_OPERATE_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "sgen.hpp"
#include "check.hpp"
#include "print.hpp"

#include "operate/operations.hpp"


#ifndef DISABLE_NOFUNCS_TEST


class NOFUNCS : public testify::fuzz_test {};


using namespace testutils;


TEST_F(NOFUNCS, MultipleFunc_B000)
{
	nnet::tensorshape shape = random_def_shape(this);
	size_t n = shape.n_elems();
	std::vector<double> argument = get_double(n, "argument", {-1, 1});
	nnet::varptr leaf = nnet::constant::get<double>(argument, shape);

	nnet::varptr res = abs(leaf);
	nnet::varptr res2 = abs(leaf);
	EXPECT_EQ(res.get(), res2.get());
}


TEST_F(NOFUNCS, NullUnar_B001)
{
	
}


TEST_F(NOFUNCS, NullBinar_B002)
{
	
}


#endif /* DISABLE_NOFUNCS_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
