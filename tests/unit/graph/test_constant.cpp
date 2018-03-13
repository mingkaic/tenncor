//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include "gtest/gtest.h"

#include "sgen.hpp"
#include "check.hpp"
#include "mock_observer.hpp"

#include "graph/leaf/constant.hpp"


#ifndef DISABLE_CONSTANT_TEST


class CONSTANT : public testify::fuzz_test {};


using namespace testutils;


// covers constant: constructor, set
TEST_F(CONSTANT, Constructor_C000)
{
	nnet::tensorshape shape = random_def_shape(this, {2, 5});
	size_t n = shape.n_elems();
	double c = get_double(1, "c")[0];

	// defined shape
	std::vector<double> v = get_double(n, "v");
	std::vector<double> v2 = get_double(n / 2, "v2");
	std::vector<double> v3 = get_double(n * 1.5, "v3");

	nnet::constant* res = nnet::constant::get(c);
	nnet::constant* res2 = nnet::constant::get(v, shape);
	nnet::constant* res3 = nnet::constant::get(v2, shape);
	nnet::constant* res4 = nnet::constant::get(v3, shape);

	std::vector<double> r1 = nnet::expose<double>(res);
	std::vector<double> r2 = nnet::expose<double>(res2);
	std::vector<double> r3 = nnet::expose<double>(res3);
	std::vector<double> r4 = nnet::expose<double>(res4);
	ASSERT_EQ(1, r1.size());
	EXPECT_EQ(c, r1[0]);

	ASSERT_EQ(n, r2.size());
	ASSERT_EQ(n, r3.size());
	ASSERT_EQ(n, r4.size());
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(v[i], r2[i]);
		EXPECT_EQ(v2[i % v2.size()], r3[i]);
		EXPECT_EQ(v3[i], r4[i]);
	}

	nnet::tensorshape cshape = res2->get_tensor()->get_shape();
	nnet::tensorshape cshape2 = res3->get_tensor()->get_shape();
	nnet::tensorshape cshape3 = res4->get_tensor()->get_shape();
	EXPECT_TRUE(tensorshape_equal(shape, cshape)) <<
		sprintf("expecting shape %p, got %p", &shape, &cshape);
	EXPECT_TRUE(tensorshape_equal(shape, cshape2)) <<
		sprintf("expecting shape %p, got %p", &shape, &cshape2);
	EXPECT_TRUE(tensorshape_equal(shape, cshape3)) <<
		sprintf("expecting shape %p, got %p", &shape, &cshape3);

	delete res;
	delete res2;
	delete res3;
	delete res4;
}


// covers constant: clone and move
TEST_F(CONSTANT, CopyNMove_C001)
{
	double c = get_double(1, "c")[0];
	nnet::tensorshape shape = random_def_shape(this);

	size_t n = shape.n_elems();
	std::vector<double> v = get_double(get_int(1, "v.size", {0.5*n, 1.5*n})[0], "v");

	nnet::constant* res = nnet::constant::get(c);
	nnet::constant* res2 = nnet::constant::get(v, shape);

	EXPECT_EQ(nullptr, res->clone());
	EXPECT_EQ(nullptr, res2->clone());
	EXPECT_EQ(nullptr, res->move());
	EXPECT_EQ(nullptr, res2->move());

	delete res;
	delete res2;
}


// covers constant: get_leaves
TEST_F(CONSTANT, GetLeaves_C002)
{
	double c = get_double(1, "c")[0];
	nnet::constant* res = nnet::constant::get(c);

	std::unordered_set<const nnet::inode*> leafset = res->get_leaves();
	ASSERT_EQ(1, leafset.size());
	EXPECT_EQ(res, *(leafset.begin())) << "res constant not found in leafset";

	delete res;
}


// covers constant: get_tensor, expose
TEST_F(CONSTANT, GetTensor_C003)
{
	double c = get_double(1, "c")[0];
	nnet::tensorshape shape = random_def_shape(this);

	size_t n = shape.n_elems();
	std::vector<double> v = get_double(get_int(1, "v.size", {0.5*n, 1.5*n})[0], "v");

	nnet::constant* res = nnet::constant::get(c);
	nnet::constant* res2 = nnet::constant::get(v, shape);

	nnet::tensor* ten = res->get_tensor();
	nnet::tensor* ten2 = res2->get_tensor();
	ASSERT_TRUE(ten->has_data()) <<
		sprintf("scalar constant %f failed to initialize with data", c);
	ASSERT_TRUE(ten2->has_data()) <<
		sprintf("scalar constant %vf failed to initialize with data", &v);
	std::vector<double> vec = nnet::expose<double>(ten);
	std::vector<double> vec2 = nnet::expose<double>(ten2);
	std::vector<double> vec3 = nnet::expose<double>(res);
	std::vector<double> vec4 = nnet::expose<double>(res2);

	ASSERT_EQ(1, vec.size());
	EXPECT_EQ(c, vec[0]);
	ASSERT_EQ(1, vec3.size());
	EXPECT_EQ(c, vec3[0]);

	ASSERT_EQ(n, vec2.size());
	ASSERT_EQ(n, vec4.size());
	for (size_t i = 0; i < vec2.size(); ++i)
	{
		EXPECT_EQ(v[i % v.size()], vec2[i]);
		EXPECT_EQ(v[i % v.size()], vec4[i]);
	}
	
	delete res;
	delete res2;
}


// covers constant: derive
TEST_F(CONSTANT, Derive_C004)
{
	double c = get_double(1, "c")[0];
	nnet::constant* res = nnet::constant::get(c);
	nnet::constant* res2 = nnet::constant::get(c+1);

	nnet::varptr g1 = res->derive(nullptr);
	nnet::varptr g2 = res->derive(res);
	nnet::varptr g3 = res->derive(res2);

	EXPECT_EQ(nullptr, g1.get());
	EXPECT_EQ(nullptr, g2.get());
	EXPECT_EQ(nullptr, g3.get());

	delete res;
	delete res2;
}


// covers constant: death_on_noparent
TEST_F(CONSTANT, SelfDestruct_C005)
{
	double c = get_double(1, "c")[0];
	nnet::constant* res = nnet::constant::get(c);
	mock_observer* mconn = new mock_observer({res});
	delete mconn;
	// memory leak if res is not destroyed
}


// covers constant: operator ==, operator !=
TEST_F(CONSTANT, ScalarEqual_C006)
{
	double c = get_double(1, "c", {-91234, 92342})[0];
	nnet::tensorshape shape = random_def_shape(this);

	size_t n = shape.n_elems();
	std::vector<double> v = get_double(get_int(1, "v.size", {0.5*n, 1.5*n})[0], "v");

	nnet::constant* res = nnet::constant::get(c);
	nnet::constant* res2 = nnet::constant::get(v, shape);

	EXPECT_TRUE(*res == c) <<
		sprintf("scalar constant %f does not equal %f", c, c);
	EXPECT_FALSE(*res == (c + 1)) <<
		sprintf("scalar constant %f equals %f", c, c + 1);
	EXPECT_FALSE(*res != c) <<
		sprintf("scalar constant %f does not equal %f", c, c);
	EXPECT_TRUE(*res != (c + 1)) <<
		sprintf("scalar constant %f equals %f", c, c + 1);

	EXPECT_FALSE(*res2 == c) <<
		sprintf("non-scalar constant %vf successfully == compares with %f", &v, c);
	EXPECT_FALSE(*res2 == (c + 1)) <<
		sprintf("non-scalar constant %vf successfully == compares with %f", &v, c + 1);
	EXPECT_FALSE(*res2 != c) <<
		sprintf("non-scalar constant %vf successfully != compares with %f", &v, c);
	EXPECT_FALSE(*res2 != (c + 1)) <<
		sprintf("non-scalar constant %vf successfully != compares with %f", &v, c + 1);

	delete res;
	delete res2;
}


#endif /* DISABLE_CONSTANT_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
