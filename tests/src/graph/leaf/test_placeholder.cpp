//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_LEAF_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "tests/include/mocks/mock_node.h"
#include "tests/include/mocks/mock_connector.h"
#include "tests/include/utils/util_test.h"
#include "tests/include/utils/fuzz.h"

#include "include/graph/leaf/placeholder.hpp"


#ifndef DISABLE_PLACEHOLDER_TEST


class PLACEHOLDER : public FUZZ::fuzz_test {};


TEST_F(PLACEHOLDER, Constructor_G000)
{
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0]);
	tensorshape shape = random_def_shape(this);

	placeholder place(shape, tenncor::tensor_proto::DOUBLE_T, label1);
	std::vector<double> raw = get_double(shape.n_elems(), "raw");

	EXPECT_FALSE(place.good_status());
}


TEST_F(PLACEHOLDER, Copy_G001)
{
	std::vector<size_t> strns = get_int(2, "strns", {14, 29});
	std::string label1 = get_string(strns[0]);
	std::string label2 = get_string(strns[1]);
	tensorshape shape = random_def_shape(this);

	placeholder assign(std::vector<size_t>{1});
	placeholder place(shape, tenncor::tensor_proto::DOUBLE_T, label1);
	std::vector<double> raw = get_double(shape.n_elems(), "raw");
	place = raw;
	placeholder* pcpy = place.clone();
	assign = place;

	std::vector<double> cpyout = expose<double>(pcpy);
	std::vector<double> assout = expose<double>(&assign);

	size_t n = raw.size();
	ASSERT_EQ(cpyout.size(), n);
	ASSERT_EQ(assout.size(), n);
	for (size_t i = 0; i < n; i++)
	{
		EXPECT_EQ(raw[i], cpyout[i]);
		EXPECT_EQ(raw[i], assout[i]);
	}

	// check re-assignment after cloning
	std::vector<double> raw2 = get_double(n, "raw2");
	placeholder assign2(std::vector<size_t>{1});
	placeholder uninit(shape, tenncor::tensor_proto::DOUBLE_T, label2);
	placeholder* uninitcpy = uninit.clone();
	assign2 = uninit;

	// copy of uninitialized placeholders should still be able to initialize
	*pcpy = raw2;
	assign = raw2;
	// copy of initialized placeholders should be able to initialize
	*uninitcpy = raw2;
	assign2 = raw2;

	cpyout = expose<double>(pcpy);
	assout = expose<double>(&assign);
	std::vector<double> cpy2out = expose<double>(uninitcpy);
	std::vector<double> ass2out = expose<double>(&assign2);

	ASSERT_EQ(cpyout.size(), n);
	ASSERT_EQ(assout.size(), n);
	ASSERT_EQ(cpy2out.size(), n);
	ASSERT_EQ(ass2out.size(), n);
	for (size_t i = 0; i < n; i++)
	{
		EXPECT_EQ(raw2[i], cpyout[i]);
		EXPECT_EQ(raw2[i], assout[i]);
		EXPECT_EQ(raw2[i], cpy2out[i]);
		EXPECT_EQ(raw2[i], ass2out[i]);
	}

	delete pcpy;
	delete uninitcpy;
}


TEST_F(PLACEHOLDER, Move_G001)
{
	placeholder assign(std::vector<size_t>{1});

	std::vector<size_t> strns = get_int(2, "strns", {14, 29});
	std::string label1 = get_string(strns[0]);
	std::string label2 = get_string(strns[1]);
	tensorshape shape = random_def_shape(this);

	placeholder place(shape, tenncor::tensor_proto::DOUBLE_T, label1);
	std::vector<double> raw = get_double(shape.n_elems(), "raw");

	size_t n = raw.size();
	place = raw;
	placeholder* pmv = place.move();

	std::vector<double> mvout = expose<double>(pmv);
	ASSERT_EQ(mvout.size(), n);

	for (size_t i = 0; i < n; i++)
	{
		EXPECT_EQ(raw[i], mvout[i]);
	}

	EXPECT_EQ(nullptr, place.eval());

	assign = std::move(*pmv);

	std::vector<double> assout = expose<double>(&assign);
	ASSERT_EQ(assout.size(), n);

	EXPECT_EQ(nullptr, pmv->eval());

	for (size_t i = 0; i < n; i++)
	{
		EXPECT_EQ(raw[i], assout[i]);
	}

	delete pmv;
}


TEST_F(PLACEHOLDER, AssignRaw_G002)
{
	mocker::usage_.clear();
	std::vector<size_t> strns = get_int(3, "strns", {14, 29});
	std::string label1 = get_string(strns[0]);
	std::string label2 = get_string(strns[1]);
	std::string label3 = get_string(strns[2]);
	tensorshape shape = random_def_shape(this);
	tensorshape part = make_partial(this, shape.as_list());

	placeholder place(shape, tenncor::tensor_proto::DOUBLE_T, label1);
	placeholder place2(part, tenncor::tensor_proto::DOUBLE_T, label2);
	std::vector<double> raw = get_double(shape.n_elems(), "raw");

	mock_connector conn({&place}, label3);
	conn.inst_ = "conn";

	EXPECT_FALSE(place.good_status());
	place = raw;
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::update1", 1));

	EXPECT_TRUE(place.good_status());
	const tensor_double* placer = dynamic_cast<const tensor_double*>(place.eval());
	EXPECT_TRUE(placer->is_alloc());
	EXPECT_TRUE(tensorshape_equal(shape, placer->get_shape()));
	std::vector<double> out = placer->expose();
	for (size_t i = 0, n = out.size(); i < n; i++)
	{
		EXPECT_EQ(raw[i], out[i]);
	}

	// partial placeholders may guess shape for input vector
	// place2 will succeed since place2 is made partial from initial shape
	place2 = raw;
	EXPECT_TRUE(place2.good_status());
	const tensor_double* placer2 = dynamic_cast<const tensor_double*>(place2.eval());
	EXPECT_TRUE(placer2->is_alloc());
	out = placer2->expose();
	for (size_t i = 0, n = out.size(); i < n; i++)
	{
		EXPECT_EQ(raw[i], out[i]);
	}
}


TEST_F(PLACEHOLDER, AssignTensor_G003)
{
	mocker::usage_.clear();
	std::vector<size_t> strns = get_int(2, "strns", {14, 29});
	std::string label1 = get_string(strns[0]);
	std::string label2 = get_string(strns[1]);
	tensorshape shape = random_def_shape(this);

	placeholder place(shape, tenncor::tensor_proto::DOUBLE_T, label1);

	double c = get_double(1, "c")[0];
	const_init cinit(c);
	tensor_double rawtens(shape);
	tensor_double* rawtenptr = &rawtens;
	cinit(*rawtenptr);

	mock_connector conn({&place}, label2);
	conn.inst_ = "conn";

	EXPECT_FALSE(place.good_status());
	place = rawtens;
	EXPECT_TRUE(mocker::EXPECT_CALL("conn::update1", 1));
	EXPECT_FALSE(rawtens.is_alloc());

	EXPECT_TRUE(place.good_status());
	const tensor_double* placer = dynamic_cast<const tensor_double*>(place.eval());
	EXPECT_TRUE(placer->is_alloc());
	EXPECT_TRUE(tensorshape_equal(shape, placer->get_shape()));
	std::vector<double> out = placer->expose();
	for (size_t i = 0, n = out.size(); i < n; i++)
	{
		EXPECT_EQ(c, out[i]);
	}
}


TEST_F(PLACEHOLDER, GetLeaf_G004)
{
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	tensorshape shape = random_def_shape(this);
	mock_node exposer;

	placeholder place(shape, tenncor::tensor_proto::DOUBLE_T, label1);

	varptr zaro = exposer.expose_leaf(&place, nullptr);
	EXPECT_TRUE(expose<double>(zaro)[0] == 0.0);
}


#endif /* DISABLE_PLACEHOLDER_TEST */


#endif /* DISABLE_LEAF_MODULE_TESTS */
