//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_LEAF_MODULE_TESTS

#include <algorithm>

#include "tests/include/mocks/mock_leaf.h"


#ifndef DISABLE_LEAF_TEST


class LEAF : public FUZZ::fuzz_test {};


// covers ileaf
// copy constructor and assignment
TEST_F(LEAF, Copy_C000)
{
	mock_leaf assign(this, "");
	mock_leaf assign2(this, "");
	mock_leaf assign3(this, "");
	
	const_init<double> cinit(get_int(1, "cinit.scalar")[0]);

	std::vector<size_t> strlens = get_int(3, "strlens", {14, 29});
	std::string label1 = get_string(strlens[0], "label1");
	std::string label2 = get_string(strlens[1], "label2");
	std::string label3 = get_string(strlens[2], "label3");
	tensorshape comp = random_def_shape(this);
	tensorshape part = make_partial(this, comp.as_list());
	mock_leaf res(comp, label1);
	mock_leaf res2(part, label2);
	mock_leaf uninit(comp, label3);
	res.set_good();
	res.mock_init_data(cinit);

	bool initstatus = res.good_status();
	bool initstatus2 = res2.good_status();
	bool initstatus3 = uninit.good_status();
	const tensor<double>* init_data = res.eval();
	const tensor<double>* init_data2 = res2.eval(); // res2 is not good

	ileaf* cpy = res.clone();
	ileaf* cpy2 = res2.clone();
	ileaf* cpy3 = uninit.clone();
	assign = res;
	assign2 = res2;
	assign3 = uninit;

	bool cpystatus = cpy->good_status();
	bool cpystatus2 = cpy2->good_status();
	bool cpystatus3 = cpy3->good_status();
	const tensor<double>* cpy_data = cpy->eval();
	const tensor<double>* cpy_data2 = cpy2->eval();

	bool assignstatus = assign.good_status();
	bool assignstatus2 = assign2.good_status();
	bool assignstatus3 = assign3.good_status();
	const tensor<double>* assign_data = assign.eval();
	const tensor<double>* assign_data2 = assign2.eval();

	EXPECT_EQ(initstatus, cpystatus);
	EXPECT_EQ(initstatus, assignstatus);
	EXPECT_EQ(initstatus2, cpystatus2);
	EXPECT_EQ(initstatus2, assignstatus2);
	EXPECT_EQ(initstatus3, cpystatus3);
	EXPECT_EQ(initstatus3, assignstatus3);
	EXPECT_NE(initstatus, cpystatus2);
	EXPECT_NE(initstatus, assignstatus2);
	EXPECT_NE(initstatus2, cpystatus);
	EXPECT_NE(initstatus2, assignstatus);

	// no point in checking memory of uninitialized data
	// expect deep copy
	EXPECT_NE(init_data, cpy_data);
	EXPECT_NE(init_data, assign_data);
	EXPECT_EQ(nullptr, init_data2);
	EXPECT_EQ(nullptr, cpy_data2);
	EXPECT_EQ(nullptr, assign_data2);
	ASSERT_NE(nullptr, init_data);
	ASSERT_NE(nullptr, cpy_data);
	ASSERT_NE(nullptr, assign_data);
	ASSERT_TRUE(tensorshape_equal(init_data->get_shape(), cpy_data->get_shape()));
	ASSERT_TRUE(tensorshape_equal(init_data->get_shape(), assign_data->get_shape()));
	// we're checking tensor copy over
	// data isn't initialized at this point, so we're exposing garabage.
	// regardless, deep copy would still copy over memory content since
	// [IMPORTANT!] tensor has no initialization data and we just pretended that status is good
	ASSERT_TRUE(init_data->is_alloc());
	ASSERT_TRUE(cpy_data->is_alloc());
	ASSERT_TRUE(assign_data->is_alloc());
	std::vector<double> idata = init_data->expose();
	std::vector<double> cdata = cpy_data->expose();
	std::vector<double> adata = assign_data->expose();

	EXPECT_TRUE(std::equal(idata.begin(), idata.end(), cdata.begin()));
	EXPECT_TRUE(std::equal(idata.begin(), idata.end(), adata.begin()));

	delete cpy;
	delete cpy2;
	delete cpy3;
}


// covers ileaf
// move constructor and assignment
TEST_F(LEAF, Move_C000)
{
	mock_leaf assign(this, "");
	mock_leaf assign2(this, "");
	mock_leaf assign3(this, "");
	
	const_init<double> cinit(get_int(1, "cinit.scalar")[0]);

	std::vector<size_t> strlens = get_int(3, "strlens", {14, 29});
	std::string label1 = get_string(strlens[0], "label1");
	std::string label2 = get_string(strlens[1], "label2");
	std::string label3 = get_string(strlens[2], "label3");
	tensorshape comp = random_def_shape(this);
	tensorshape part = make_partial(this, comp.as_list());
	mock_leaf res(comp, label1);
	mock_leaf res2(part, label2);
	mock_leaf uninit(comp, label3);
	res.set_good();
	res.mock_init_data(cinit);

	bool initstatus = res.good_status();
	bool initstatus2 = res2.good_status();
	bool initstatus3 = uninit.good_status();
	const tensor<double>* init_data = res.eval();
	const tensor<double>* init_data2 = res2.eval();
	const tensor<double>* init_data3 = uninit.eval();

	ileaf* mv = res.move();
	ileaf* mv2 = res2.move();
	ileaf* mv3 = uninit.move();

	bool mvstatus = mv->good_status();
	bool mvstatus2 = mv2->good_status();
	bool mvstatus3 = mv3->good_status();
	// ensure shallow copy
	const tensor<double>* mv_data = mv->eval();
	const tensor<double>* mv_data2 = mv2->eval();
	const tensor<double>* mv_data3 = mv3->eval();
	EXPECT_EQ(init_data, mv_data);
	EXPECT_EQ(init_data2, mv_data2);
	EXPECT_EQ(init_data3, mv_data3);
	EXPECT_NE(init_data2, mv_data);
	EXPECT_NE(init_data, mv_data2);
	EXPECT_EQ(nullptr, res.eval());
	EXPECT_EQ(nullptr, res2.eval());
	EXPECT_EQ(nullptr, uninit.eval());

	mock_leaf* mmv = static_cast<mock_leaf*>(mv);
	mock_leaf* mmv2 = static_cast<mock_leaf*>(mv2);
	mock_leaf* mmv3 = static_cast<mock_leaf*>(mv3);
	assign = std::move(*mmv);
	assign2 = std::move(*mmv2);
	assign3 = std::move(*mmv3);

	bool assignstatus = assign.good_status();
	bool assignstatus2 = assign2.good_status();
	bool assignstatus3 = assign3.good_status();
	// ensure shallow copy
	const tensor<double>* assign_data = assign.eval();
	const tensor<double>* assign_data2 = assign2.eval();
	const tensor<double>* assign_data3 = assign3.eval();
	EXPECT_EQ(mv_data, assign_data);
	EXPECT_EQ(mv_data2, assign_data2);
	EXPECT_EQ(mv_data3, assign_data3);
	EXPECT_NE(mv_data, assign_data2);
	EXPECT_NE(mv_data2, assign_data);
	EXPECT_EQ(nullptr, mv->eval());
	EXPECT_EQ(nullptr, mv2->eval());
	EXPECT_EQ(nullptr, mv3->eval());

	EXPECT_EQ(initstatus, mvstatus);
	EXPECT_EQ(initstatus, assignstatus);
	EXPECT_EQ(initstatus2, mvstatus2);
	EXPECT_EQ(initstatus2, assignstatus2);
	EXPECT_EQ(initstatus3, mvstatus3);
	EXPECT_EQ(initstatus3, assignstatus3);
	EXPECT_NE(initstatus, mvstatus2);
	EXPECT_NE(initstatus, assignstatus2);
	EXPECT_NE(initstatus2, mvstatus);
	EXPECT_NE(initstatus2, assignstatus);

	delete mv;
	delete mv2;
	delete mv3;
}


// covers ileaf get_shape
TEST_F(LEAF, GetShape_C001)
{
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	std::string label2 = get_string(get_int(1, "label2.size", {14, 29})[0], "label2");
	tensorshape shape1 = random_shape(this);
	tensorshape shape2 = random_shape(this);
	mock_leaf res(shape1, label1);
	mock_leaf res2(shape2, label2);

	tensorshape rshape1 = res.get_shape();
	tensorshape rshape2 = res2.get_shape();
	EXPECT_TRUE(tensorshape_equal(shape1, rshape1));
	EXPECT_TRUE(tensorshape_equal(shape2, rshape2));

	EXPECT_EQ(tensorshape_equal(shape1, shape2),
		tensorshape_equal(shape2, rshape1));
	EXPECT_EQ(tensorshape_equal(shape1, shape2),
		tensorshape_equal(shape1, rshape2));
}


// covers ileaf eval
TEST_F(LEAF, GetEval_C002)
{
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	std::string label2 = get_string(get_int(1, "label2.size", {14, 29})[0], "label2");
	tensorshape comp = random_def_shape(this);
	tensorshape part = make_partial(this, comp.as_list());
	mock_leaf res(comp, label1);
	mock_leaf res2(part, label2);
	// pretend they're both good (initialized)
	res.set_good();
	res2.set_good();

	const tensor<double>* rout = res.eval();
	const tensor<double>* r2out = res2.eval();

	ASSERT_NE(nullptr, rout);
	ASSERT_NE(nullptr, r2out);
	EXPECT_TRUE(tensorshape_equal(comp, rout->get_shape()));
	EXPECT_TRUE(tensorshape_equal(part, r2out->get_shape()));
	EXPECT_TRUE(rout->is_alloc());
	EXPECT_FALSE(r2out->is_alloc());
}


// covers ileaf good_status
TEST_F(LEAF, GoodStatus_C003)
{
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0]);
	std::string label2 = get_string(get_int(1, "label2.size", {14, 29})[0]);
	tensorshape comp = random_def_shape(this);
	tensorshape part = make_partial(this, comp.as_list());
	mock_leaf res(comp, label1);
	mock_leaf res2(part, label2);
	res.set_good();

	EXPECT_TRUE(res.good_status());
	EXPECT_FALSE(res2.good_status());
}


// todo: implement
//C004 - reading a valid tensor_proto should initialize the leaf


TEST_F(LEAF, GetLeaves_G005)
{
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	tensorshape shape = random_def_shape(this);

	mock_leaf res(shape, label1);

	std::unordered_set<ileaf*> leafset = res.get_leaves();
	EXPECT_TRUE(leafset.end() != leafset.find(&res));
}


#endif /* DISABLE_LEAF_TEST */


#endif /* DISABLE_LEAF_MODULE_TESTS */
