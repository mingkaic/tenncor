//
// Created by Mingkai Chen on 2017-09-12.
//

#ifndef DISABLE_CONNECTOR_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "tests/include/utils/util_test.h"
#include "tests/include/utils/fuzz.h"

#include "include/graph/leaf/constant.hpp"
#include "include/graph/connector/immutable/generator.hpp"


#ifndef DISABLE_GENERATOR_TEST


class GENERATOR : public FUZZ::fuzz_test {};


TEST_F(GENERATOR, Copy_J000)
{
	tensorshape shape = random_def_shape(this);
	double c = get_double(1, "c", {1, 17})[0];

	std::vector<double> cdata(shape.n_elems(), 0);
	constant* con = constant::get(cdata, shape);
	const_init<double> cinit(c);
	rand_uniform<double> rinit(-12, -2);

	generator* gen_assign = generator::get(con, rinit);

	generator* gen = generator::get(con, cinit);

	generator* gen_cpy = gen->clone();

	*gen_assign = *gen;

	EXPECT_TRUE(tensorshape_equal(gen->get_shape(), shape));
	EXPECT_TRUE(tensorshape_equal(gen_cpy->get_shape(), shape));
	EXPECT_TRUE(tensorshape_equal(gen_assign->get_shape(), shape));

	std::vector<double> gvec = nnet::expose<double>(gen);
	std::vector<double> gvec_cpy = nnet::expose<double>(gen_cpy);
	std::vector<double> gvec_assign = nnet::expose<double>(gen_assign);
	std::all_of(gvec.begin(), gvec.end(), [c](double e) { return e == c; });
	std::all_of(gvec_cpy.begin(), gvec_cpy.end(), [c](double e) { return e == c; });
	std::all_of(gvec_assign.begin(), gvec_assign.end(), [c](double e) { return e == c; });

	delete con;
}


TEST_F(GENERATOR, Move_J000)
{
	tensorshape shape = random_def_shape(this);
	double c = get_double(1, "c", {1, 17})[0];

	std::vector<double> cdata(shape.n_elems(), 0);
	constant* con = constant::get(cdata, shape);
	const_init<double> cinit(c);
	rand_uniform<double> rinit(-12, -2);

	generator* gen_assign = generator::get(con, rinit);

	generator* gen = generator::get(con, cinit);
	EXPECT_TRUE(tensorshape_equal(gen->get_shape(), shape));
	std::vector<double> gvec = nnet::expose<double>(gen);
	std::all_of(gvec.begin(), gvec.end(), [c](double e) { return e == c; });

	generator* gen_mv = gen->move();
	EXPECT_TRUE(tensorshape_equal(gen_mv->get_shape(), shape));
	std::vector<double> gvec_mv = nnet::expose<double>(gen_mv);
	std::all_of(gvec_mv.begin(), gvec_mv.end(), [c](double e) { return e == c; });

	*gen_assign = std::move(*gen_mv);
	EXPECT_TRUE(tensorshape_equal(gen_assign->get_shape(), shape));
	std::vector<double> gvec_assign = nnet::expose<double>(gen_assign);
	std::all_of(gvec_assign.begin(), gvec_assign.end(), [c](double e) { return e == c; });

	delete con;
	delete gen;
	delete gen_mv;
}


TEST_F(GENERATOR, ShapeDep_J001)
{
	tensorshape shape = random_def_shape(this);
	double c = get_double(1, "c", {1, 17})[0];
	std::vector<double> cdata(shape.n_elems(), 0);
	constant* con = constant::get(cdata, shape);
	const_init<double> cinit(c);

	generator* gen = generator::get(con, cinit);
	EXPECT_TRUE(tensorshape_equal(gen->get_shape(), shape));

	delete con;
}


TEST_F(GENERATOR, Derive_J002)
{
	tensorshape shape = random_def_shape(this);
	double c = get_double(1, "c", {1, 17})[0];
	std::vector<double> cdata(shape.n_elems(), 0);
	constant* con = constant::get(cdata, shape);
	const_init<double> cinit(c);

	generator* gen = generator::get(con, cinit);

	inode* wan = nullptr;
	gen->temporary_eval(gen, wan);
	constant* wanc = dynamic_cast<constant*>(wan);
	ASSERT_NE(nullptr, wanc);
	EXPECT_TRUE(*wanc == 1.0);

	varptr zaro = gen->derive(con);
	varptr wan2 = gen->derive(gen);
	constant* zaroc = dynamic_cast<constant*>(zaro.get());
	ASSERT_NE(nullptr, zaroc);
	EXPECT_TRUE(*zaroc == 0.0);
	constant* wanc2 = dynamic_cast<constant*>(wan2.get());
	ASSERT_NE(nullptr, wanc2);
	EXPECT_TRUE(*wanc2 == 1.0);

	delete con;
	delete wan;
}


#endif /* DISABLE_GENERATOR_TEST */


#endif /* DISABLE_CONNECTOR_MODULE_TESTS */
