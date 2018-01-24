//
// Created by Mingkai Chen on 2017-03-10.
//

#ifndef DISABLE_TENSOR_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "tests/include/utils/fuzz.h"
#include "tests/include/utils/util_test.h"
#include "tests/include/mocks/mock_itensor.h"
// avoid mock tensor to prevent random initialization

#include "include/tensor/tensor_double.hpp"
#include "include/tensor/tensor_handler.hpp"


using namespace nnet;


#ifndef DISABLE_HANDLER_TEST


class HANDLER : public FUZZ::fuzz_test {};


// todo: remove this bad practice, maybe deterministically mark shapes and data if necessary
static double SUPERMARK = 1;


struct forward_mark : public tens_template<double>
{
	forward_mark (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
	tens_template(dest, srcs) {}

	virtual void action (void)
	{
		size_t n = this->dest_.second.n_elems();
		for (size_t i = 0; i < n; ++i)
		{
			this->dest_.first[i] = SUPERMARK;
		}
	}
};


static itens_actor* marked_forward (out_wrapper<void>& dest, 
	std::vector<in_wrapper<void> >& srcs, tenncor::tensor_proto::tensor_t type)
{
	return new forward_mark(dest, srcs);
}


static tensorshape shaper (std::vector<tensorshape> args)
{
	std::vector<size_t> s0 = args[0].as_list();
	std::vector<size_t> s1 = args[1].as_list();
	std::vector<size_t> compress;
	for (size_t i = 0, n = std::min(s0.size(), s1.size()); i < n; i++)
	{
		compress.push_back(std::min(s0[i], s1[i]));
	}
	return compress;
}


// cover transfer_function
// operator ()
TEST_F(HANDLER, Transfer_D000)
{
	tensorshape c1 = random_def_shape(this);
	tensorshape c2 = random_def_shape(this);
	tensorshape resshape = shaper({c1, c2});
	mock_itensor arg1(this, c1);
	mock_itensor arg2(this, c2);
	tensor_double good(resshape);
	std::vector<const itensor*> args = { &arg1, &arg2 };

	actor_func tf(adder);
	itens_actor* actor = tf(good, args);
	actor->action();
	std::vector<double> d1 = arg1.expose();
	std::vector<double> d2 = arg2.expose();
	std::vector<double> res = good.expose();
	size_t n = c1.n_elems();
	size_t m = c2.n_elems();
	size_t l = resshape.n_elems();
	for (size_t i = 0, k = std::min(std::min(n, m), l); i < k; i++)
	{
		EXPECT_EQ(res[i], d1[i] + d2[i]);
	}

	delete actor;
}


TEST_F(HANDLER, DISABLED_Assign_)
{
	// todo: implement + add to behavior.txt
}


// cover const_init
// operator ()
TEST_F(HANDLER, Constant_D001)
{
	double scalar = get_double(1, "scalar")[0];
	const_init ci(scalar);
	tensorshape shape = random_def_shape(this);
	tensor_double block(shape);
	ci(block);

	std::vector<double> v = block.expose();

	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_EQ(scalar, v[i]);
	}
}


// cover rand_uniform, rand_normal
// operator ()
TEST_F(HANDLER, Random_D002)
{
	double lo = get_double(1, "lo", {127182, 12921231412323})[0];
	double hi = lo+1;
	double high = get_double(1, "high", {lo*2, lo*3+50})[0];
	double mean = get_double(1, "mean", {-13, 23})[0];
	double variance = get_double(1, "variance", {1, 32})[0];
	rand_uniform ri1(lo, hi);
	rand_uniform ri2(lo, high);
	rand_normal rn(mean, variance);
	tensorshape shape = random_def_shape(this);
	tensor_double block1(shape);
	tensor_double block2(shape);
	tensor_double block3(shape);
	ri1(block1);
	ri2(block2);
	rn(block3);

	std::vector<double> v1 = block1.expose();
	std::vector<double> v2 = block2.expose();
	std::vector<double> v3 = block3.expose();

	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(lo, v1[i]);
		EXPECT_GE(hi, v1[i]);
	}
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(lo, v2[i]);
		EXPECT_GE(high, v2[i]);
	}
	// assert shape has n_elem of [17, 7341]
	// the mean of vnorm is definitely within variance / 2 of the normal
	double norm_mean = std::accumulate(v3.begin(), v3.end(), 0) / (double) v3.size();
	EXPECT_GT(norm_mean, mean - variance);
	EXPECT_LT(norm_mean, mean + variance);
}


// cover actor_func, const_init, rand_uniform
// copy constructor and assignment
TEST_F(HANDLER, Copy_D003)
{
	SUPERMARK = 0;
	actor_func tfassign(marked_forward);
	const_init ciassign(0);
	rand_uniform riassign(0, 1);
	rand_normal niassign;

	SUPERMARK = get_double(1, "SUPERMARK", {15, 117})[0];
	double scalar = get_double(1, "scalar")[0];
	double low = get_double(1, "low", {23, 127})[0];
	double high = get_double(1, "high", {low*2, low*3+50})[0];
	double mean = get_double(1, "mean", {-13, 23})[0];
	double variance = get_double(1, "variance", {1, 32})[0];
	actor_func tf(marked_forward);
	const_init ci(scalar);
	rand_uniform ri(low, high);
	rand_normal ni(mean, variance);

	actor_func* tfcpy = tf.clone();
	const_init* cicpy = ci.clone();
	rand_uniform* ricpy = ri.clone();
	rand_normal* nicpy = ni.clone();

	tfassign = tf;
	ciassign = ci;
	riassign = ri;
	niassign = ni;

	tensorshape shape = random_def_shape(this);
	tensor_double tscalar(0);
	tensor_double tblock(shape);
	tensor_double tblock_norm(shape);
	tensor_double ttransf(std::vector<size_t>{(size_t) SUPERMARK});
	std::vector<const itensor*> empty_args;
	itens_actor* actor = (*tfcpy)(ttransf, empty_args);
	actor->action();
	std::vector<double> transfv = ttransf.expose();
	EXPECT_EQ((size_t)SUPERMARK, transfv.size());
	EXPECT_EQ(SUPERMARK, transfv[0]);

	(*cicpy)(tscalar);
	EXPECT_EQ(scalar, tscalar.expose()[0]);

	(*ricpy)(tblock);
	std::vector<double> v = tblock.expose();
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(low, v[i]);
		EXPECT_GE(high, v[i]);
	}

	(*nicpy)(tblock_norm);
	std::vector<double> vnorm = tblock_norm.expose();
	// assert shape has n_elem of [17, 7341]
	// the mean of vnorm is definitely within variance / 2 of the normal
	double norm_mean = std::accumulate(vnorm.begin(), vnorm.end(), 0) / (double) vnorm.size();
	EXPECT_GT(norm_mean, mean - variance);
	EXPECT_LT(norm_mean, mean + variance);

	tensor_double tscalar2(0);
	tensor_double tblock2(shape);
	tensor_double tblock_norm2(shape);
	tensor_double ttransf2(std::vector<size_t>{(size_t) SUPERMARK});
	itens_actor* actor2 = tfassign(ttransf2, empty_args);
	actor2->action();
	std::vector<double> transfv2 = ttransf2.expose();
	EXPECT_EQ((size_t)SUPERMARK, transfv2.size());
	EXPECT_EQ(SUPERMARK, transfv2[0]);

	ciassign(tscalar2);
	EXPECT_EQ(scalar, tscalar2.expose()[0]);

	riassign(tblock2);
	std::vector<double> v2 = tblock2.expose();
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(low, v2[i]);
		EXPECT_GE(high, v2[i]);
	}

	niassign(tblock_norm2);
	std::vector<double> vnorm2 = tblock_norm2.expose();
	// assert shape has n_elem of [17, 7341]
	// the mean of vnorm is definitely within variance / 2 of the normal
	double norm_mean2 = std::accumulate(vnorm2.begin(), vnorm2.end(), 0) / (double) vnorm2.size();
	EXPECT_GT(norm_mean2, mean - variance);
	EXPECT_LT(norm_mean2, mean + variance);

	itensor_handler* interface_ptr = &tf;
	itensor_handler* resptr = interface_ptr->clone();
	EXPECT_NE(nullptr, dynamic_cast<actor_func*>(resptr));
	delete resptr;

	interface_ptr = &ci;
	resptr = interface_ptr->clone();
	EXPECT_NE(nullptr, dynamic_cast<const_init*>(resptr));
	delete resptr;

	interface_ptr = &ri;
	resptr = interface_ptr->clone();
	EXPECT_NE(nullptr, dynamic_cast<rand_uniform*>(resptr));
	delete resptr;

	delete tfcpy;
	delete cicpy;
	delete ricpy;
	delete nicpy;

	delete actor;
	delete actor2;
}


// cover actor_func, const_init, rand_uniform
// move constructor and assignment
TEST_F(HANDLER, Move_D003)
{
	SUPERMARK = 0;
	actor_func tfassign(marked_forward);
	const_init ciassign(0);
	rand_uniform riassign(0, 1);
	rand_normal niassign;

	SUPERMARK = get_double(1, "SUPERMARK", {119, 221})[0];
	double scalar = get_double(1, "scalar")[0];
	double low = get_double(1, "low", {23, 127})[0];
	double high = get_double(1, "high", {low*2, low*3+50})[0];
	double mean = get_double(1, "mean", {-13, 23})[0];
	double variance = get_double(1, "variance", {1, 32})[0];
	actor_func tf(marked_forward);
	const_init ci(scalar);
	rand_uniform ri(low, high);
	rand_normal ni(mean, variance);

	actor_func* tfmv = tf.move();
	const_init* cimv = ci.move();
	rand_uniform* rimv = ri.move();
	rand_normal* nimv = ni.move();

	tensorshape shape = random_def_shape(this);
	tensor_double tscalar(0);
	tensor_double tblock(shape);
	tensor_double tblock_norm(shape);
	tensor_double ttransf(std::vector<size_t>{(size_t) SUPERMARK});
	std::vector<const itensor*> empty_args;
	itens_actor* actor = (*tfmv)(ttransf, empty_args);
	actor->action();
	std::vector<double> transfv = ttransf.expose();
	EXPECT_EQ((size_t)SUPERMARK, transfv.size());
	EXPECT_EQ(SUPERMARK, transfv[0]);

	(*cimv)(tscalar);
	EXPECT_EQ(scalar, tscalar.expose()[0]);

	(*rimv)(tblock);
	std::vector<double> v = tblock.expose();
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(low, v[i]);
		EXPECT_GE(high, v[i]);
	}

	(*nimv)(tblock_norm);
	std::vector<double> vnorm = tblock_norm.expose();
	// assert shape has n_elem of [17, 7341]
	// the mean of vnorm is definitely within variance / 2 of the normal
	double norm_mean = std::accumulate(vnorm.begin(), vnorm.end(), 0) / (double) vnorm.size();
	EXPECT_GT(norm_mean, mean - variance);
	EXPECT_LT(norm_mean, mean + variance);

	tfassign = std::move(*tfmv);
	ciassign = std::move(*cimv);
	riassign = std::move(*rimv);
	niassign = std::move(*nimv);

	tensor_double tscalar2(0);
	tensor_double tblock2(shape);
	tensor_double tblock_norm2(shape);
	tensor_double ttransf2(std::vector<size_t>{(size_t) SUPERMARK});
	itens_actor* actor2 = tfassign(ttransf2, empty_args);
	actor2->action();
	std::vector<double> transfv2 = ttransf2.expose();
	EXPECT_EQ((size_t)SUPERMARK, transfv2.size());
	EXPECT_EQ(SUPERMARK, transfv2[0]);

	ciassign(tscalar2);
	EXPECT_EQ(scalar, tscalar2.expose()[0]);

	riassign(tblock2);
	std::vector<double> v2 = tblock2.expose();
	for (size_t i = 0, n = shape.n_elems(); i < n; i++)
	{
		EXPECT_LE(low, v2[i]);
		EXPECT_GE(high, v2[i]);
	}

	niassign(tblock_norm2);
	std::vector<double> vnorm2 = tblock_norm2.expose();
	// assert shape has n_elem of [17, 7341]
	// the mean of vnorm is definitely within variance of the normal
	double norm_mean2 = std::accumulate(vnorm2.begin(), vnorm2.end(), 0) / (double) vnorm2.size();
	EXPECT_GT(norm_mean2, mean - variance);
	EXPECT_LT(norm_mean2, mean + variance);

	delete tfmv;
	delete cimv;
	delete rimv;
	delete nimv;

	delete actor;
	delete actor2;
}


#endif /* DISABLE_HANDLER_TEST */

#endif /* DISABLE_TENSOR_MODULE_TESTS */
