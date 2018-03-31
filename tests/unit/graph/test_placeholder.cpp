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

#include "graph/placeholder.hpp"


#ifndef DISABLE_PLACEHOLDER_TEST


class PLACEHOLDER : public testutils::fuzz_test
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


// covers variable: constructors
TEST_F(PLACEHOLDER, Constructor_E000)
{
	std::string label1 = get_string(get_int(1, "label1.size", {14, 29})[0], "label1");
	nnet::tensorshape shape = random_def_shape(this);

	nnet::placeholder place(shape, label1);
	std::vector<double> raw = get_double(shape.n_elems(), "raw");

	nnet::tensor* ten = place.get_tensor();
	ASSERT_NE(nullptr, ten);

	EXPECT_FALSE(ten->has_data()) <<
		"tensor ten has data";
	nnet::tensorshape gotshape = ten->get_shape();
	EXPECT_TRUE(tensorshape_equal(shape, gotshape)) <<
		testutils::sprintf("expecting shape %p, got %p", &shape, &gotshape);

	place = raw;
	EXPECT_TRUE(ten->has_data()) <<
		"tensor ten does not have data";

	std::vector<double> data = nnet::expose<double>(&place);
	EXPECT_TRUE(std::equal(raw.begin(), raw.end(), data.begin())) <<
		testutils::sprintf("expecting %vf, got %vf", &raw, &data);
}


// covers placeholder: clone
TEST_F(PLACEHOLDER, Copy_E001)
{
	std::vector<size_t> strns = get_int(2, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label2");
	std::vector<size_t> clist = random_def_shape(this);
	nnet::tensorshape shape = clist;
	nnet::tensorshape badshape = make_incompatible(clist);

	nnet::placeholder assign(badshape, label1);
	mock_observer mconn({&assign});

	nnet::placeholder place(shape, label2);

	nnet::tensor* pten = place.get_tensor();
	nnet::tensor* aten = assign.get_tensor();
	ASSERT_NE(nullptr, pten);
	ASSERT_NE(nullptr, aten);
	ASSERT_FALSE(pten->has_data()) <<
		"tensor pten has data";
	ASSERT_FALSE(aten->has_data()) <<
		"tensor aten has data";

	std::vector<double> raw = get_double(shape.n_elems(), "raw");
	place = raw;
	ASSERT_TRUE(pten->has_data()) <<
		"tensor pten does not have data";

	// clone
	nnet::placeholder* pcpy = place.clone();
	nnet::tensor* cten = pcpy->get_tensor();
	ASSERT_NE(nullptr, cten);
	ASSERT_TRUE(cten->has_data()) <<
		"tensor cten does not have data";
	EXPECT_NE(pten, cten);

	std::vector<double> cpyout = nnet::expose<double>(pcpy);
	EXPECT_TRUE(std::equal(raw.begin(), raw.end(), cpyout.begin())) <<
		testutils::sprintf("expecting %vf, got %vf", &raw, &cpyout);

	// assign
	assign = place;
	nnet::tensor* aten2 = assign.get_tensor();
	ASSERT_NE(nullptr, aten2);
	ASSERT_TRUE(aten2->has_data()) <<
		"tensor aten2 does not have data";
	EXPECT_NE(aten, aten2);
	EXPECT_NE(pten, aten2);

	std::vector<double> assout = nnet::expose<double>(&assign);
	EXPECT_TRUE(std::equal(raw.begin(), raw.end(), assout.begin())) <<
		testutils::sprintf("expecting %vf, got %vf", &raw, &assout);

	size_t n_updates = testify::mocker::get_usage(&mconn, "update2");
	optional<std::string> updateval = testify::mocker::get_value(&mconn, "update2");
	EXPECT_EQ(1, n_updates);
	ASSERT_TRUE((bool) updateval) <<
		"mconn update2 value is not found";
	EXPECT_STREQ("UPDATE", updateval->c_str());

	delete pcpy;
}


// covers placeholder: move
TEST_F(PLACEHOLDER, Move_E001)
{
	std::vector<size_t> strns = get_int(2, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label2");
	std::vector<size_t> clist = random_def_shape(this);
	nnet::tensorshape shape = clist;
	nnet::tensorshape badshape = make_incompatible(clist);

	nnet::placeholder assign(badshape, label1);
	mock_observer mconn({&assign});

	nnet::placeholder place(shape, label2);

	nnet::tensor* pten = place.get_tensor();
	nnet::tensor* aten = assign.get_tensor();
	ASSERT_NE(nullptr, pten);
	ASSERT_NE(nullptr, aten);
	ASSERT_FALSE(pten->has_data()) <<
		"tensor pten has data";
	ASSERT_FALSE(aten->has_data()) <<
		"tensor aten has data";

	std::vector<double> raw = get_double(shape.n_elems(), "raw");
	place = raw;
	ASSERT_TRUE(pten->has_data()) <<
		"tensor pten does not have data";

	// move
	nnet::placeholder* pmv = place.move();
	nnet::tensor* mten = pmv->get_tensor();
	ASSERT_NE(nullptr, mten);
	ASSERT_TRUE(mten->has_data()) <<
		"tensor mten does not have data";
	EXPECT_EQ(pten, mten);

	std::vector<double> mvout = nnet::expose<double>(pmv);
	EXPECT_TRUE(std::equal(raw.begin(), raw.end(), mvout.begin())) <<
		testutils::sprintf("expecting %vf, got %vf", &raw, &mvout);
	EXPECT_EQ(nullptr, place.get_tensor());

	// assign
	assign = std::move(*pmv);
	nnet::tensor* aten2 = assign.get_tensor();
	EXPECT_NE(nullptr, aten2);
	ASSERT_TRUE(aten2->has_data()) <<
		"tensor aten2 does not have data";
	EXPECT_NE(aten, aten2);
	EXPECT_EQ(pten, aten2);

	std::vector<double> assout = nnet::expose<double>(&assign);
	EXPECT_TRUE(std::equal(raw.begin(), raw.end(), assout.begin())) <<
		testutils::sprintf("expecting %vf, got %vf", &raw, &assout);
	EXPECT_EQ(nullptr, pmv->get_tensor());

	size_t n_updates = testify::mocker::get_usage(&mconn, "update2");
	optional<std::string> updateval = testify::mocker::get_value(&mconn, "update2");
	EXPECT_EQ(1, n_updates);
	ASSERT_TRUE((bool) updateval) <<
		"mconn update2 value is not found";
	EXPECT_STREQ("UPDATE", updateval->c_str());

	delete pmv;
}


// covers placeholder: get_leaves
TEST_F(PLACEHOLDER, GetLeaves_E002)
{
	std::string label = get_string(get_int(1, "label.size", {14, 29})[0], "label");
	nnet::tensorshape shape = random_def_shape(this);
	nnet::placeholder res(shape, label);

	std::unordered_set<const nnet::inode*> leafset = res.get_leaves();
	ASSERT_EQ(1, leafset.size());
	EXPECT_EQ(&res, *(leafset.begin())) << "res placeholder not found in leafset";
}


// covers placeholder: get_tensor
TEST_F(PLACEHOLDER, GetTensor_E003)
{
	std::string label = get_string(get_int(1, "label.size", {14, 29})[0], "label");
	nnet::tensorshape shape = random_def_shape(this);
	nnet::placeholder res(shape, label);
	std::vector<double> raw = get_double(shape.n_elems(), "raw");

	nnet::tensor* ten = res.get_tensor();
	EXPECT_FALSE(ten->has_data()) <<
		"tensor ten has data";
	res = raw;
	EXPECT_TRUE(ten->has_data()) <<
		"tensor ten does not have data";
}


// covers placeholder: derive
TEST_F(PLACEHOLDER, Derive_E004)
{
	std::string label = get_string(get_int(1, "label.size", {14, 29})[0], "label");
	nnet::tensorshape shape = random_def_shape(this);
	nnet::placeholder res(shape, label);
	nnet::placeholder res2(shape, label);

	nnet::varptr g1 = res.derive(nullptr);
	nnet::varptr g2 = res.derive(&res);
	nnet::varptr g3 = res.derive(&res2);

	EXPECT_EQ(nullptr, g1.get());
	EXPECT_EQ(nullptr, g2.get());
	EXPECT_EQ(nullptr, g3.get());
}


// covers placeholder: operator = (std::vector<T>)
TEST_F(PLACEHOLDER, AssignRaw_E005)
{
	std::vector<size_t> strns = get_int(3, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label2");
	std::string label3 = get_string(strns[2], "label3");

	std::vector<size_t> clist = random_def_shape(this);
	nnet::tensorshape shape = clist;
	nnet::tensorshape part = make_partial(this, clist);

	nnet::placeholder place(part, label1);
	nnet::placeholder place2(shape, label2);
	nnet::placeholder place3(shape, label3);
	size_t exactdata = shape.n_elems();
	std::vector<double> raw = get_double(exactdata, "raw");
	size_t lowerdata = 1;
	if (exactdata >= 3)
	{
		lowerdata = exactdata - get_int(1,
			"exactdata - lowerdata", {1, exactdata-1})[0];
	}
	std::vector<double> badraw = get_double(lowerdata, "badraw");

	nnet::tensor* ten = place.get_tensor();
	nnet::tensor* ten2 = place2.get_tensor();

	mock_observer mconn({&place});
	mock_observer mconn2({&place2});

	// assign with guess shape to fit (should be shape)
	EXPECT_FALSE(ten->has_data()) <<
		"tensor ten has data";
	nnet::tensorshape gotshape = ten->get_shape();
	EXPECT_TRUE(tensorshape_equal(part, gotshape)) <<
		testutils::sprintf("expecting shape %p, got %p", &shape, &gotshape);
	place = raw;
	EXPECT_TRUE(ten->has_data()) <<
		"tensor ten does not have data";
	gotshape = ten->get_shape();
	EXPECT_EQ(shape.n_elems(), gotshape.n_elems());

	std::vector<double> data1 = nnet::expose<double>(&place);
	EXPECT_TRUE(std::equal(raw.begin(), raw.end(), data1.begin())) <<
		testutils::sprintf("expecting %vf, got %vf", &raw, &data1);

	size_t n_updates = testify::mocker::get_usage(&mconn, "update2");
	optional<std::string> updateval = testify::mocker::get_value(&mconn, "update2");
	EXPECT_EQ(1, n_updates);
	ASSERT_TRUE((bool) updateval) <<
		"mconn update2 value is not found";
	EXPECT_STREQ("UPDATE", updateval->c_str());

	// regular assign
	EXPECT_FALSE(ten2->has_data()) <<
		"tensor ten2 has data";
	place2 = raw;
	EXPECT_TRUE(ten2->has_data()) <<
		"tensor ten2 does not have data";
	gotshape = ten2->get_shape();
	EXPECT_TRUE(tensorshape_equal(shape, gotshape)) <<
		testutils::sprintf("expecting shape %p, got %p", &shape, &gotshape);

	size_t n_updates2 = testify::mocker::get_usage(&mconn2, "update2");
	optional<std::string> updateval2 = testify::mocker::get_value(&mconn2, "update2");
	EXPECT_EQ(1, n_updates2);
	ASSERT_TRUE((bool) updateval2) <<
		"mconn2 update2 value is not found";
	EXPECT_STREQ("UPDATE", updateval2->c_str());

	std::vector<double> data2 = nnet::expose<double>(&place2);
	EXPECT_TRUE(std::equal(raw.begin(), raw.end(), data2.begin())) <<
		testutils::sprintf("expecting %vf, got %vf", &raw, &data2);

	// bad assignment (failure to guess shape)
	EXPECT_THROW((place3 = badraw), std::logic_error);

	// initialized assign
	std::vector<double> raw2 = get_double(exactdata, "raw2");
	place = raw2;

	std::vector<double> data3 = nnet::expose<double>(&place);
	EXPECT_TRUE(std::equal(raw2.begin(), raw2.end(), data3.begin())) <<
		testutils::sprintf("expecting %vf, got %vf", &raw2, &data3);

	EXPECT_EQ(2, testify::mocker::get_usage(&mconn, "update2"));

	// bad initialized assignment
	EXPECT_THROW((place2 = badraw), std::logic_error);
}


// covers placeholder: operator = (tensor)
TEST_F(PLACEHOLDER, AssignTensor_E006)
{
	std::vector<size_t> strns = get_int(2, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label1");

	std::vector<size_t> clist = random_def_shape(this);
	nnet::tensorshape shape = clist;
	nnet::tensorshape part = make_partial(this, clist);
	nnet::tensorshape badshape = make_incompatible(clist);
	size_t n = shape.n_elems();

	double c = get_double(1, "c")[0];
	nnet::const_init ci;
	ci.set<double>(c);

	nnet::tensor good(shape);
	nnet::tensor bad(badshape);

	nnet::placeholder place(part, label1);
	nnet::placeholder place2(shape, label2);

	nnet::tensor* ten = place.get_tensor();
	nnet::tensor* ten2 = place2.get_tensor();

	mock_observer mconn({&place});
	mock_observer mconn2({&place2});

	EXPECT_FALSE(ten->has_data()) <<
		"tensor ten has data";
	EXPECT_FALSE(ten2->has_data()) <<
		"tensor ten2 has data";

	// expect fail
	EXPECT_THROW((place = good), std::exception);
	EXPECT_THROW((place2 = good), std::exception);
	EXPECT_FALSE(ten->has_data()) <<
		"tensor ten has data";
	EXPECT_FALSE(ten2->has_data()) <<
		"tensor ten2 has data";

	size_t n_updates = testify::mocker::get_usage(&mconn, "update2");
	size_t n_updates2 = testify::mocker::get_usage(&mconn2, "update2");
	EXPECT_EQ(0, n_updates);
	EXPECT_EQ(0, n_updates2);

	ASSERT_TRUE(good.read_from(ci)) <<
		testutils::sprintf("failed to read constant %f", c);
	ASSERT_TRUE(bad.read_from(ci)) <<
		testutils::sprintf("failed to read constant %f", c);

	ASSERT_TRUE(good.has_data()) <<
		"good doesn't have data";
	ASSERT_TRUE(bad.has_data()) <<
		"bad doesn't have data";

	// incomatible assign
	place = bad;
	place2 = bad;
	EXPECT_FALSE(ten->has_data()) <<
		"tensor ten has data";
	EXPECT_FALSE(ten2->has_data()) <<
		"tensor ten2 has data";

	n_updates = testify::mocker::get_usage(&mconn, "update2");
	n_updates2 = testify::mocker::get_usage(&mconn2, "update2");
	EXPECT_EQ(0, n_updates);
	EXPECT_EQ(0, n_updates2);

	// assign self
	place = *ten;
	place2 = *ten2;
	EXPECT_FALSE(ten->has_data()) <<
		"tensor ten has data";
	EXPECT_FALSE(ten2->has_data()) <<
		"tensor ten2 has data";

	n_updates = testify::mocker::get_usage(&mconn, "update2");
	n_updates2 = testify::mocker::get_usage(&mconn2, "update2");
	EXPECT_EQ(0, n_updates);
	EXPECT_EQ(0, n_updates2);

	place = good;
	place2 = good;
	ASSERT_TRUE(ten->has_data()) <<
		"tensor ten does not have data";
	ASSERT_TRUE(ten2->has_data()) <<
		"tensor ten2 does not have data";

	std::vector<double> pdata = nnet::expose<double>(&place);
	std::vector<double> pdata2 = nnet::expose<double>(&place2);

	EXPECT_EQ(n, pdata.size());
	EXPECT_EQ(n, pdata2.size());

	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(c, pdata[i]);
		EXPECT_EQ(c, pdata2[i]);
	}

	n_updates = testify::mocker::get_usage(&mconn, "update2");
	n_updates2 = testify::mocker::get_usage(&mconn2, "update2");
	optional<std::string> updateval = testify::mocker::get_value(&mconn, "update2");
	optional<std::string> updateval2 = testify::mocker::get_value(&mconn2, "update2");
	EXPECT_EQ(1, n_updates);
	EXPECT_EQ(1, n_updates2);
	ASSERT_TRUE((bool) updateval) <<
		"mconn update2 value is not found";
	ASSERT_TRUE((bool) updateval2) <<
		"mconn2 update2 value is not found";
	EXPECT_STREQ("UPDATE", updateval->c_str());
	EXPECT_STREQ("UPDATE", updateval2->c_str());
}


#endif /* DISABLE_PLACEHOLDER_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
