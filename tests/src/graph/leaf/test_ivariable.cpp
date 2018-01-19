//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_LEAF_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "tests/include/mocks/mock_ivariable.h"
#include "tests/include/utils/fuzz.h"


#ifndef DISABLE_IVARIABLE_TEST


class IVARIABLE : public FUZZ::fuzz_test {};


TEST_F(IVARIABLE, Copy_E000)
{
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	tensorshape shape = random_def_shape(this);
	double c = get_double(1, "c")[0];

	const_init* cinit = new const_init(c);

	mock_ivariable assign(shape, nullptr, "");
	mock_ivariable assign2(shape, nullptr, "");
	mock_ivariable noinit(shape, nullptr, label1);
	mock_ivariable inited(shape, cinit, label1);

	mock_ivariable* cpy = static_cast<mock_ivariable*>(noinit.clone());
	mock_ivariable* cpy2 = static_cast<mock_ivariable*>(inited.clone());
	assign = noinit;
	assign2 = inited;

	initializer* noi = noinit.get_initializer();
	EXPECT_EQ(noi, cpy->get_initializer());
	EXPECT_EQ(noi, assign.get_initializer());
	EXPECT_EQ(nullptr, noi);

	tensor_double ct(std::vector<size_t>{1});
	tensor_double ct2(std::vector<size_t>{1});
	initializer* ci = cpy2->get_initializer();
	initializer* ai = assign2.get_initializer();
	tensor_double* ctptr = &ct;
	tensor_double* ct2ptr = &ct2;
	(*ci)(*ctptr);
	(*ai)(*ct2ptr);
	EXPECT_EQ(c, ct.expose()[0]);
	EXPECT_EQ(c, ct2.expose()[0]);

	delete cpy;
	delete cpy2;
}


TEST_F(IVARIABLE, Move_E000)
{
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	tensorshape shape = random_def_shape(this);
	double c = get_double(1, "c")[0];

	const_init* cinit = new const_init(c);

	mock_ivariable assign(shape, nullptr, "");
	mock_ivariable assign2(shape, nullptr, "");
	mock_ivariable noinit(shape, nullptr, label1);
	mock_ivariable inited(shape, cinit, label1);

	initializer* noi = noinit.get_initializer();
	initializer* ii = inited.get_initializer();

	EXPECT_EQ(nullptr, noi);

	mock_ivariable* mv = static_cast<mock_ivariable*>(noinit.move());
	mock_ivariable* mv2 = static_cast<mock_ivariable*>(inited.move());

	EXPECT_EQ(noi, mv->get_initializer());
	tensor_double ct(std::vector<size_t>{1});
	initializer* mi = mv2->get_initializer();
	tensor_double* ctptr = &ct;
	(*mi)(*ctptr);
	EXPECT_EQ(c, ct.expose()[0]);
	EXPECT_EQ(ii, mi);
	EXPECT_EQ(nullptr, inited.get_initializer());
	EXPECT_EQ(nullptr, inited.eval());

	assign = std::move(*mv);
	assign2 = std::move(*mv2);

	EXPECT_EQ(noi, assign.get_initializer());
	tensor_double ct2(std::vector<size_t>{1});
	initializer* ai = assign2.get_initializer();
	tensor_double* ct2ptr = &ct2;
	(*ai)(*ct2ptr);
	EXPECT_EQ(c, ct2.expose()[0]);
	EXPECT_EQ(ii, ai);
	EXPECT_EQ(nullptr, mv2->get_initializer());
	EXPECT_EQ(nullptr, mv2->eval());

	delete mv;
	delete mv2;
}


TEST_F(IVARIABLE, Initialize_E001)
{
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	tensorshape shape = random_def_shape(this);
	double c = get_double(1, "c")[0];
	const_init* cinit = new const_init(c);

	mock_ivariable noinit(shape, nullptr, label1);
	mock_ivariable inited(shape, cinit, label1);

	EXPECT_FALSE(noinit.can_init());
	EXPECT_TRUE(inited.can_init());
}


TEST_F(IVARIABLE, GetGradient_E002)
{
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	tensorshape shape = random_def_shape(this);
	double c = get_double(1, "c")[0];
	const_init* cinit = new const_init(c);

	mock_ivariable noinit(shape, nullptr, label1);
	mock_ivariable inited(shape, cinit, label1);

	const tensor_double* wun = dynamic_cast<const tensor_double*>(noinit.derive(&noinit)->eval());
	const tensor_double* wuntoo = dynamic_cast<const tensor_double*>(inited.derive(&inited)->eval());
	const tensor_double* zaro = dynamic_cast<const tensor_double*>(noinit.derive(&inited)->eval());
	const tensor_double* zarotoo = dynamic_cast<const tensor_double*>(inited.derive(&noinit)->eval());

	EXPECT_EQ((size_t) 1, wun->n_elems());
	EXPECT_EQ((size_t) 1, wuntoo->n_elems());
	EXPECT_EQ((size_t) 1, zaro->n_elems());
	EXPECT_EQ((size_t) 1, zarotoo->n_elems());
	EXPECT_EQ(1.0, wun->expose()[0]);
	EXPECT_EQ(1.0, wuntoo->expose()[0]);
	EXPECT_EQ(0.0, zaro->expose()[0]);
	EXPECT_EQ(0.0, zarotoo->expose()[0]);
}


#endif /* DISABLE_IVARIABLE_TEST */


#endif /* DISABLE_LEAF_MODULE_TESTS */
