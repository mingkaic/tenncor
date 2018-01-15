//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_LEAF_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "tests/include/mocks/mock_node.h"
#include "tests/include/mocks/mock_connector.h"
#include "tests/include/util_test.h"
#include "tests/include/fuzz.h"

#include "include/graph/leaf/constant.hpp"


using namespace nnet;


#ifndef DISABLE_CONSTANT_TEST


class CONSTANT : public FUZZ::fuzz_test {};


// covers constant
// scalar constructor, vector constructor
TEST_F(CONSTANT, Constructor_D000)
{
	tensorshape shape;
	tensorshape part;
	size_t n;
	size_t pn = 1;
	// re-roll until we get partial with n_known> 1
	while (pn <= 1)
	{
	 // keep resetting FUZZ logger
		shape = random_def_shape(this, 2, 5); // constraint rank to force relatively non-one shapes
		part = make_partial(this, shape.as_list());
		n = shape.n_elems();
		pn = part.n_known();
	}
	double c = get_double(1, "c")[0];

	// defined shape
	std::vector<double> v = get_double(n, "v");
	std::vector<double> v2 = get_double(n / 2, "v2");
	std::vector<double> v3 = get_double(n * 1.5, "v3");

	// partially defined shape
	std::vector<double> pv = get_double(pn, "pv");
	std::vector<double> pv2 = get_double(pn * 0.6, "pv2");
	std::vector<double> pv3 = get_double(pn * 1.5, "pv3");

	constant* res = constant::get(c);
	constant* res2 = constant::get(v, shape);
	constant* res3 = constant::get(v2, shape);
	constant* res4 = constant::get(v3, shape);
	constant* res5 = constant::get(pv, part);
	constant* res6 = constant::get(pv2, part);
	constant* res7 = constant::get(pv3, part);

	EXPECT_TRUE(res->good_status()); // scalars are initialized

	std::vector<double> r2 = expose<double>(res2);
	std::vector<double> r3 = expose<double>(res3);
	std::vector<double> r4 = expose<double>(res4);
	EXPECT_EQ(n, r2.size());
	EXPECT_EQ(n, r3.size());
	EXPECT_EQ(n, r4.size());
	for (size_t i = 0; i < n; i++)
	{
		EXPECT_EQ(v[i], r2[i]);
		if (i < n/2)
		{
			EXPECT_EQ(v2[i], r3[i]);
		}
		else
		{
			EXPECT_EQ((double) 0, r3[i]);
		}
		EXPECT_EQ(v3[i], r4[i]);
	}
	EXPECT_TRUE(tensorshape_equal(shape, res2->get_shape()));
	EXPECT_TRUE(tensorshape_equal(shape, res3->get_shape()));
	EXPECT_TRUE(tensorshape_equal(shape, res4->get_shape()));

	std::vector<double> r5 = expose<double>(res5);
	std::vector<double> r6 = expose<double>(res6);
	std::vector<double> r7 = expose<double>(res7);

	EXPECT_EQ(pn, r5.size());
	EXPECT_EQ(pn, r6.size());
	size_t pv3n = pv3.size();
	size_t v7n = r7.size();
	ASSERT_LT(pn, pv3n);
	ASSERT_LT(pv3n, v7n);
	for (size_t i = 0; i < pn; i++)
	{
		EXPECT_EQ(pv[i], r5[i]);
		if (i < pv2.size())
		{
			EXPECT_EQ(pv2[i], r6[i]);
		}
		else
		{
			EXPECT_EQ((double) 0, r6[i]);
		}
		EXPECT_EQ(pv3[i], r7[i]);
	}
	size_t i = pn;
	for (; i < pv3n; i++)
	{
		EXPECT_EQ(pv3[i], r7[i]);
	}
	for (; i < v7n; i++)
	{
		EXPECT_EQ((double) 0, r7[i]);
	}

	// the shapes of res5 to 7 should be compatible with part
	EXPECT_TRUE(part.is_compatible_with(res5->get_shape()));
	EXPECT_TRUE(part.is_compatible_with(res3->get_shape()));
	EXPECT_TRUE(part.is_compatible_with(res4->get_shape()));

	delete res;
	delete res2;
	delete res3;
	delete res4;
	delete res5;
	delete res6;
	delete res7;
}


// covers constant
// clone and move
TEST_F(CONSTANT, CopyNMove_D001)
{
	double c = get_double(1, "c")[0];
	tensorshape shape = random_def_shape(this);
	tensorshape part = make_partial(this, shape.as_list());

	size_t n = shape.n_elems();
	size_t pn = part.n_known();
	// defined shape
	std::vector<double> v = get_double(get_int(1, "v.size", {0.5*n, 1.5*n})[0], "v");
	// partially defined shape
	std::vector<double> pv = get_double(get_int(1, "pv.size", {0.5*pn, 1.5*pn})[0], "pv");

	constant* res = constant::get(c);
	constant* res2 = constant::get(v, shape);
	constant* res3 = constant::get(pv, part);

	EXPECT_EQ(nullptr, res->clone());
	EXPECT_EQ(nullptr, res2->clone());
	EXPECT_EQ(nullptr, res3->clone());
	EXPECT_EQ(nullptr, res->move());
	EXPECT_EQ(nullptr, res2->move());
	EXPECT_EQ(nullptr, res3->move());

	delete res;
	delete res2;
	delete res3;
}


// covers constant derive
TEST_F(CONSTANT, GetGradient_D002)
{
	double c = get_double(1, "c")[0];
	constant* res = constant::get(c);
	constant* res2 = constant::get(c+1);

	const tensor<double>* g1 = res->derive(nullptr)->eval();
	const tensor<double>* g2 = res->derive(res)->eval();
	const tensor<double>* g3 = res->derive(res2)->eval();

	std::vector<double> gres = g1->expose();
	std::vector<double> gres1 = g2->expose();
	std::vector<double> gres2 = g3->expose();

	ASSERT_EQ((size_t) 1, gres.size());
	ASSERT_EQ((size_t) 1, gres1.size());
	ASSERT_EQ((size_t) 1, gres2.size());

	EXPECT_EQ((double) 0, gres[0]);
	EXPECT_EQ((double) 0, gres1[0]);
	EXPECT_EQ((double) 0, gres2[0]);

	delete res;
	delete res2;
}


// covers constant get_gradient
TEST_F(CONSTANT, GetLeaf_D003)
{
	double c = get_double(1, "c")[0];
	constant* res = constant::get(c);
	mock_node exposer;

	varptr zaro = exposer.expose_leaf(res, nullptr);
	EXPECT_TRUE(expose<double>(zaro)[0] == 0.0);

	delete res;
}


// covers constant death_on_noparent
TEST_F(CONSTANT, SelfDestruct_D004)
{
	double c = get_double(1, "c")[0];
	constant* res = constant::get(c); // managed
	constant* res2 = constant::get(c); // unmanaged
	res->be_managed();

	mock_connector* mconn = new mock_connector({res, res2}, "");
	delete mconn;

	EXPECT_NE(nullptr, res->eval());
	delete res;
}


// verifies data status
TEST_F(CONSTANT, Allocated_D005)
{
	double c = get_double(1, "c")[0];
	tensorshape shape = random_def_shape(this);
	tensorshape part = make_partial(this, shape.as_list());

	size_t n = shape.n_elems();
	size_t pn = part.n_known();
	// defined shape
	std::vector<double> v = get_double(get_int(1, "v.size", {0.5*n, 1.5*n})[0], "v");
	// partially defined shape
	if (1 == pn) pn = 2;
	std::vector<double> pv = get_double(get_int(1, "pv.size", {0.5*pn, 1.5*pn})[0], "pv");

	constant* res = constant::get(c);
	constant* res2 = constant::get(v, shape);
	constant* res3 = constant::get(pv, part);

	const tensor<double>* t1 = res->eval();
	const tensor<double>* t2 = res2->eval();
	const tensor<double>* t3 = res3->eval();

	EXPECT_TRUE(t1->is_alloc());
	EXPECT_TRUE(t2->is_alloc());
	EXPECT_TRUE(t3->is_alloc());

	delete res;
	delete res2;
	delete res3;
}


#endif /* DISABLE_CONSTANT_TEST */


#endif /* DISABLE_LEAF_MODULE_TESTS */
