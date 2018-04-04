//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzz.hpp"
#include "sgen.hpp"
#include "check.hpp"
#include "print.hpp"
#include "mock_observer.hpp"

#include "graph/constant.hpp"


#ifndef DISABLE_CONSTANT_TEST


class CONSTANT : public testutils::fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		testutils::fuzz_test::TearDown();
		testify::mocker::clear();
	}
};


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

	nnet::varptr res = nnet::constant::get<double>(c);
	nnet::varptr same = nnet::constant::get<double>(c);
	EXPECT_EQ(res.get(), same.get());

	nnet::varptr res2 = nnet::constant::get<double>(v, shape);
	nnet::varptr res3 = nnet::constant::get<double>(v2, shape);
	nnet::varptr res4 = nnet::constant::get<double>(v3, shape);

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
	EXPECT_SHAPEQ(shape, cshape);
	EXPECT_SHAPEQ(shape,  cshape2);
	EXPECT_SHAPEQ(shape,  cshape3);
}


// covers constant: clone and move
TEST_F(CONSTANT, CopyNMove_C001)
{
	double c = get_double(1, "c")[0];
	nnet::tensorshape shape = random_def_shape(this);

	size_t n = shape.n_elems();
	std::vector<double> v = get_double(get_int(1, "v.size", {0.5*n, 1.5*n})[0], "v");

	nnet::varptr res = nnet::constant::get<double>(c);
	nnet::varptr res2 = nnet::constant::get<double>(v, shape);

	EXPECT_EQ(nullptr, res->clone());
	EXPECT_EQ(nullptr, res2->clone());
	EXPECT_EQ(nullptr, res->move());
	EXPECT_EQ(nullptr, res2->move());
}


// covers constant: get_leaves
TEST_F(CONSTANT, GetLeaves_C002)
{
	double c = get_double(1, "c")[0];
	nnet::varptr res = nnet::constant::get<double>(c);

	std::unordered_set<const nnet::inode*> leafset = res->get_leaves();
	ASSERT_EQ(1, leafset.size());
	EXPECT_EQ(res, *(leafset.begin())) << "res constant not found in leafset";
}


// covers constant: get_tensor, expose
TEST_F(CONSTANT, GetTensor_C003)
{
	double c = get_double(1, "c")[0];
	nnet::tensorshape shape = random_def_shape(this);

	size_t n = shape.n_elems();
	std::vector<double> v = get_double(get_int(1, "v.size", {0.5*n, 1.5*n})[0], "v");

	nnet::varptr res = nnet::constant::get<double>(c);
	nnet::varptr res2 = nnet::constant::get<double>(v, shape);

	nnet::tensor* ten = res->get_tensor();
	nnet::tensor* ten2 = res2->get_tensor();
	ASSERT_TRUE(ten->has_data()) <<
		testutils::sprintf("scalar constant %f failed to initialize with data", c);
	ASSERT_TRUE(ten2->has_data()) <<
		testutils::sprintf("scalar constant %vf failed to initialize with data", &v);
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
}


// covers constant: derive
TEST_F(CONSTANT, Derive_C004)
{
	double c = get_double(1, "c")[0];
	nnet::varptr res = nnet::constant::get<double>(c);
	nnet::varptr res2 = nnet::constant::get<double>(c+1);

	nnet::varptr g1 = res->derive(nullptr);
	nnet::varptr g2 = res->derive(res);
	nnet::varptr g3 = res->derive(res2);

	EXPECT_EQ(nullptr, g1.get());
	EXPECT_EQ(nullptr, g2.get());
	EXPECT_EQ(nullptr, g3.get());
}


// covers constant: death_on_noparent
TEST_F(CONSTANT, SelfDestruct_C005)
{
	double c = get_double(1, "c")[0];
	nnet::constant* ptr;
	{
		nnet::varptr res = nnet::constant::get<double>(c);
		ptr = static_cast<nnet::constant*>(res.get());
	}
	EXPECT_FALSE(nnet::dangling(ptr)) << 
		testutils::sprintf("dangling ptr %e", (void*) ptr);
}


#endif /* DISABLE_CONSTANT_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
