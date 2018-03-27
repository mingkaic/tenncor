//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include "gtest/gtest.h"

#include "sgen.hpp"
#include "check.hpp"
#include "mock_src.hpp"
#include "mock_observer.hpp"

#include "graph/variable.hpp"


#ifndef DISABLE_VARIABLE_TEST


class VARIABLE : public testify::fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		testify::fuzz_test::TearDown();
		testify::mocker::clear();
	}
};


using namespace testutils;


// covers variable: constructors
TEST_F(VARIABLE, Constructor_D000)
{
	std::vector<size_t> strns = get_int(2, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label2");
	nnet::tensorshape shape = random_def_shape(this);
	double c = get_double(1, "c")[0];
	std::vector<double> minmax = get_double(2, "min-max", {-24, 26});
	double min = *std::min_element(minmax.begin(), minmax.end());
	double max = *std::max_element(minmax.begin(), minmax.end());

	std::shared_ptr<nnet::const_init> cinit = std::make_shared<nnet::const_init>();
	cinit->set<double>(c);
	std::shared_ptr<nnet::r_uniform_init> rinit = std::make_shared<nnet::r_uniform_init>();
	rinit->set<double>(min, max);

	nnet::variable cinitv(shape, cinit, label1);
	nnet::variable rinitv(shape, rinit, label2);

	nnet::tensor* cten = cinitv.get_tensor();
	nnet::tensor* rten = rinitv.get_tensor();
	ASSERT_NE(nullptr, cten);
	ASSERT_NE(nullptr, rten);

	EXPECT_FALSE(cten->has_data()) <<
		"tensor cten has data";
	EXPECT_FALSE(rten->has_data()) <<
		"tensor rten has data";
	nnet::tensorshape gotshape = cten->get_shape();
	EXPECT_TRUE(tensorshape_equal(shape, gotshape)) <<
		sprintf("expecting shape %p, got %p", &shape, &gotshape);
	nnet::tensorshape gotshape2 = rten->get_shape();
	EXPECT_TRUE(tensorshape_equal(shape, gotshape2)) <<
		sprintf("expecting shape %p, got %p", &shape, &gotshape2);

	cinitv.initialize();
	rinitv.initialize();

	EXPECT_TRUE(cten->has_data()) <<
		"tensor cten does not have data";
	EXPECT_TRUE(rten->has_data()) <<
		"tensor rten does not have data";
	EXPECT_EQ(DOUBLE, cten->get_type());
	EXPECT_EQ(DOUBLE, rten->get_type());
	std::vector<double> cdata = nnet::expose<double>(cten);
	std::vector<double> rdata = nnet::expose<double>(rten);
	for (double gotc : cdata)
	{
		EXPECT_EQ(c, gotc);
	}

	for (double gotr : rdata)
	{
		EXPECT_LE(min, gotr);
		EXPECT_GE(max, gotr);
	}
}


// covers variable: clone
TEST_F(VARIABLE, Copy_D001)
{
	std::vector<size_t> strns = get_int(2, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label2");
	nnet::tensorshape shape = random_def_shape(this);
	nnet::tensorshape shape2 = random_def_shape(this);
	std::shared_ptr<mock_data_src> src = std::make_shared<mock_data_src>(this);

	nnet::variable assign(shape2, nullptr, label2);
	mock_observer mconn({&assign});

	nnet::variable var(shape, src, label1);
	nnet::tensor* ten = var.get_tensor();
	ASSERT_NE(nullptr, ten);
	ASSERT_FALSE(ten->has_data()) <<
		"tensor ten has data";
	var.initialize();
	ASSERT_TRUE(ten->has_data()) <<
		"tensor ten does not have data";

	// clone
	nnet::variable* cp = var.clone();
	nnet::tensor* cpten = cp->get_tensor();
	ASSERT_NE(nullptr, cpten);
	ASSERT_TRUE(cpten->has_data()) <<
		"tensor cpten does not have data";
	EXPECT_NE(src.get(), cp->get_source().get());
	EXPECT_NE(ten, cpten);

	// assign
	assign = var;
	nnet::tensor* asten = assign.get_tensor();
	ASSERT_NE(nullptr, asten);
	ASSERT_TRUE(asten->has_data()) <<
		"tensor asten does not have data";
	EXPECT_NE(src.get(), assign.get_source().get());
	EXPECT_NE(ten, asten);

	size_t n_updates = testify::mocker::get_usage(&mconn, "update2");
	optional<std::string> updateval = testify::mocker::get_value(&mconn, "update2");
	EXPECT_EQ(1, n_updates);
	ASSERT_TRUE((bool) updateval) <<
		"mconn update2 value is not found";
	EXPECT_STREQ("UPDATE", updateval->c_str());

	delete cp;
}


// covers variable: move
TEST_F(VARIABLE, Move_D001)
{
	std::vector<size_t> strns = get_int(2, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label2");
	nnet::tensorshape shape = random_def_shape(this);
	nnet::tensorshape shape2 = random_def_shape(this);
	std::shared_ptr<mock_data_src> src = std::make_shared<mock_data_src>(this);

	nnet::variable assign(shape2, nullptr, label2);
	mock_observer mconn({&assign});

	nnet::variable var(shape, src, label1);
	nnet::tensor* ten = var.get_tensor();
	ASSERT_NE(nullptr, ten);
	ASSERT_FALSE(ten->has_data()) <<
		"tensor ten has data";
	var.initialize();
	ASSERT_TRUE(ten->has_data()) <<
		"tensor ten does not have data";

	// move
	nnet::variable* mv = var.move();
	nnet::tensor* mvten = mv->get_tensor();
	ASSERT_NE(nullptr, mvten);
	ASSERT_TRUE(mvten->has_data()) <<
		"tensor mvten does not have data";
	EXPECT_EQ(src.get(), mv->get_source().get());
	EXPECT_EQ(ten, mvten);

	EXPECT_EQ(nullptr, var.get_source().get());
	EXPECT_EQ(nullptr, var.get_tensor());

	// assign
	assign = std::move(*mv);
	nnet::tensor* asten = assign.get_tensor();
	ASSERT_NE(nullptr, asten);
	ASSERT_TRUE(asten->has_data()) <<
		"tensor asten does not have data";
	EXPECT_EQ(src.get(), assign.get_source().get());
	EXPECT_EQ(ten, asten);

	EXPECT_EQ(nullptr, mv->get_source().get());
	EXPECT_EQ(nullptr, mv->get_tensor());
	
	size_t n_updates = testify::mocker::get_usage(&mconn, "update2");
	optional<std::string> updateval = testify::mocker::get_value(&mconn, "update2");
	EXPECT_EQ(1, n_updates);
	ASSERT_TRUE((bool) updateval) <<
		"mconn update2 value is not found";;
	EXPECT_STREQ("UPDATE", updateval->c_str());

	delete mv;
}


// covers variable: get_leaves
TEST_F(VARIABLE, GetLeaves_D002)
{
	std::string label = get_string(get_int(1, "label.size", {14, 29})[0], "label");
	nnet::tensorshape shape = random_def_shape(this);
	std::shared_ptr<mock_data_src> src = std::make_shared<mock_data_src>(this);
	nnet::variable res(shape, src, label);

	std::unordered_set<const nnet::inode*> leafset = res.get_leaves();
	ASSERT_EQ(1, leafset.size());
	EXPECT_EQ(&res, *(leafset.begin())) << "res variable not found in leafset";
}


// covers variable: get_tensor
TEST_F(VARIABLE, GetTensor_D003)
{
	std::string label = get_string(get_int(1, "label.size", {14, 29})[0], "label");
	nnet::tensorshape shape = random_def_shape(this);
	std::shared_ptr<mock_data_src> src = std::make_shared<mock_data_src>(this);
	nnet::variable res(shape, src, label);

	nnet::tensor* ten = res.get_tensor();
	EXPECT_FALSE(ten->has_data()) <<
		"tensor ten has data";
	res.initialize();
	EXPECT_TRUE(ten->has_data()) <<
		"tensor ten does not have data";
}


// covers variable: derive
TEST_F(VARIABLE, Derive_D004)
{
	std::string label = get_string(get_int(1, "label.size", {14, 29})[0], "label");
	nnet::tensorshape shape = random_def_shape(this);
	double c = get_double(1, "c")[0];

	std::shared_ptr<nnet::const_init> src = std::make_shared<nnet::const_init>();
	src->set<double>(c);
	nnet::variable res(shape, src, label);
	nnet::variable res2(shape, src, label);

	nnet::varptr enul = res.derive(&res2);
	nnet::varptr enul2 = res2.derive(&res);
	nnet::varptr ewun = res.derive(&res);
	nnet::varptr ewun2 = res2.derive(&res2);

	EXPECT_EQ(nullptr, enul);
	EXPECT_EQ(nullptr, enul2);
	EXPECT_NE(ewun, ewun2);
	nnet::tensor* wten = ewun->get_tensor();
	nnet::tensor* wten2 = ewun2->get_tensor();
	ASSERT_NE(nullptr, wten);
	ASSERT_NE(nullptr, wten2);
	nnet::tensorshape gotshape = wten->get_shape();
	nnet::tensorshape gotshape2 = wten2->get_shape();
	ASSERT_TRUE(tensorshape_equal(shape, gotshape)) <<
		sprintf("expecting shape %p, got %p", &shape, &gotshape);
	ASSERT_TRUE(tensorshape_equal(shape, gotshape2)) <<
		sprintf("expecting shape %p, got %p", &shape, &gotshape2);
	std::vector<double> wunvec = nnet::expose<double>(ewun);
	std::vector<double> wunvec2 = nnet::expose<double>(ewun2);
	size_t n = shape.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(1, wunvec[i]);
		EXPECT_EQ(1, wunvec2[i]);
	}
}


// covers variable: initialize
TEST_F(VARIABLE, Initialize_D005)
{
	std::vector<size_t> strns = get_int(4, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label2");
	std::string label3 = get_string(strns[2], "label3");
	std::string label4 = get_string(strns[3], "label4");

	std::vector<size_t> clist = random_def_shape(this);
	std::vector<size_t> plist;
	std::vector<size_t> ilist;
	make_incom_partials(this, clist, plist, ilist);
	nnet::tensorshape shape = clist;
	nnet::tensorshape badshape = ilist;
	nnet::tensorshape part = plist;
	double c = get_double(1, "c")[0];
	std::vector<double> minmax = get_double(2, "min-max", {-24, 26});
	double min = *std::min_element(minmax.begin(), minmax.end());
	double max = *std::max_element(minmax.begin(), minmax.end());
	size_t n = shape.n_elems();

	std::shared_ptr<nnet::const_init> cinit = std::make_shared<nnet::const_init>();
	cinit->set<double>(c);
	std::shared_ptr<nnet::r_uniform_init> rinit = std::make_shared<nnet::r_uniform_init>();
	rinit->set<double>(min, max);

	nnet::variable cpart(part, cinit, label1);
	nnet::variable cfull(shape, cinit, label2);
	nnet::variable rpart(part, rinit, label3);
	nnet::variable rfull(shape, rinit, label4);

	mock_observer cpartobs({&cpart});
	mock_observer cfullobs({&cfull});
	mock_observer rpartobs({&rpart});
	mock_observer rfullobs({&rfull});

	nnet::tensor* cpten = cpart.get_tensor();
	nnet::tensor* cften = cfull.get_tensor();
	nnet::tensor* rpten = rpart.get_tensor();
	nnet::tensor* rften = rfull.get_tensor();

	EXPECT_FALSE(cpten->has_data()) <<
		"tensor cpten has data";
	EXPECT_FALSE(cften->has_data()) <<
		"tensor cften has data";
	EXPECT_FALSE(rpten->has_data()) <<
		"tensor rpten has data";
	EXPECT_FALSE(rften->has_data()) <<
		"tensor rften has data";

	EXPECT_FALSE(cpart.initialize()) <<
		sprintf("partial variable shape %p initialized", &part);
	EXPECT_TRUE(cfull.initialize()) <<
		sprintf("variable shape %p failed to initialize", &shape);
	EXPECT_FALSE(rpart.initialize()) <<
		sprintf("partial variable shape %p initialized", &part);
	EXPECT_TRUE(rfull.initialize()) <<
		sprintf("variable shape %p failed to initialize", &shape);

	EXPECT_FALSE(cpten->has_data()) <<
		"tensor cpten has data";
	EXPECT_TRUE(cften->has_data()) <<
		"tensor cften does not have data";
	EXPECT_FALSE(rpten->has_data()) <<
		"tensor rpten has data";
	EXPECT_TRUE(rften->has_data()) <<
		"tensor rften does not have data";
	std::vector<double> firstr = nnet::expose<double>(rften);

	size_t cp_updates = testify::mocker::get_usage(&cpartobs, "update2");
	size_t cf_updates = testify::mocker::get_usage(&cfullobs, "update2");
	size_t rp_updates = testify::mocker::get_usage(&rpartobs, "update2");
	size_t rf_updates = testify::mocker::get_usage(&rfullobs, "update2");
	optional<std::string> cf_updateval = testify::mocker::get_value(&cfullobs, "update2");
	optional<std::string> rf_updateval = testify::mocker::get_value(&rfullobs, "update2");
	EXPECT_EQ(0, cp_updates);
	EXPECT_EQ(1, cf_updates);
	EXPECT_EQ(0, rp_updates);
	EXPECT_EQ(1, rf_updates);
	ASSERT_TRUE((bool) cf_updateval) <<
		"cfullobs update2 value is not found";;
	ASSERT_TRUE((bool) rf_updateval) <<
		"rfullobs update2 value is not found";;
	EXPECT_STREQ("UPDATE", cf_updateval->c_str());
	EXPECT_STREQ("UPDATE", rf_updateval->c_str());

	EXPECT_TRUE(cpart.initialize(shape)) <<
		sprintf("cpart with shape %p failed to re-initialize with shape %p", &part, &shape);
	EXPECT_TRUE(cfull.initialize(shape)) <<
		sprintf("cfull failed to re-initialize with shape %p", &shape);
	EXPECT_TRUE(rpart.initialize(shape)) <<
		sprintf("rpart with shape %p failed to re-initialize with shape %p", &part, &shape);
	EXPECT_TRUE(rfull.initialize(shape)) <<
		sprintf("rfull failed to re-initialize with shape %p", &shape);

	EXPECT_TRUE(cpten->has_data()) <<
		"tensor cpten does not have data";
	EXPECT_TRUE(rpten->has_data()) <<
		"tensor rpten does not have data";

	cp_updates = testify::mocker::get_usage(&cpartobs, "update2");
	cf_updates = testify::mocker::get_usage(&cfullobs, "update2");
	rp_updates = testify::mocker::get_usage(&rpartobs, "update2");
	rf_updates = testify::mocker::get_usage(&rfullobs, "update2");
	optional<std::string> cp_updateval = testify::mocker::get_value(&cpartobs, "update2");
	optional<std::string> rp_updateval = testify::mocker::get_value(&rpartobs, "update2");
	EXPECT_EQ(1, cp_updates);
	EXPECT_EQ(2, cf_updates);
	EXPECT_EQ(1, rp_updates);
	EXPECT_EQ(2, rf_updates);
	ASSERT_TRUE((bool) cp_updateval) <<
		"cpartobs update2 value is not found";;
	ASSERT_TRUE((bool) rp_updateval) <<
		"rpartobs update2 value is not found";;
	EXPECT_STREQ("UPDATE", cp_updateval->c_str());
	EXPECT_STREQ("UPDATE", rp_updateval->c_str());

	std::vector<double> cpdata = nnet::expose<double>(cpten);
	std::vector<double> cfdata = nnet::expose<double>(cften);
	std::vector<double> rpdata = nnet::expose<double>(rpten);
	std::vector<double> rfdata = nnet::expose<double>(rften);

	EXPECT_FALSE(std::equal(firstr.begin(), firstr.end(), rfdata.begin())) <<
		"second initialization of random variable failed repopulate with random data";
	
	ASSERT_EQ(n, cpdata.size());
	ASSERT_EQ(n, cfdata.size());
	ASSERT_EQ(n, rpdata.size());
	ASSERT_EQ(n, rfdata.size());

	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(c, cpdata[i]);
		EXPECT_EQ(c, cfdata[i]);
		EXPECT_LE(min, rpdata[i]);
		EXPECT_GE(max, rpdata[i]);
		EXPECT_LE(min, rfdata[i]);
		EXPECT_GE(max, rfdata[i]);
	}

	EXPECT_FALSE(cpart.initialize(badshape)) <<
		sprintf("partial shape %p initialized with badshape", &part, &badshape);
	EXPECT_FALSE(cfull.initialize(badshape)) <<
		sprintf("full shape %p initialized with badshape", &shape, &badshape);
	EXPECT_FALSE(rpart.initialize(badshape)) <<
		sprintf("partial shape %p initialized with badshape", &part, &badshape);
	EXPECT_FALSE(rfull.initialize(badshape)) <<
		sprintf("full shape %p initialized with badshape", &shape, &badshape);
}


// covers variable: assign
TEST_F(VARIABLE, Assign_D006)
{
	std::vector<size_t> strns = get_int(5, "strns", {14, 29});
	std::string label1 = get_string(strns[0], "label1");
	std::string label2 = get_string(strns[1], "label2");
	std::string label3 = get_string(strns[2], "label3");
	std::string label4 = get_string(strns[3], "label4");
	std::string label5 = get_string(strns[4], "label5");

	std::vector<size_t> clist = random_def_shape(this);
	std::vector<size_t> plist;
	std::vector<size_t> ilist;
	make_incom_partials(this, clist, plist, ilist);
	nnet::tensorshape shape = clist;
	nnet::tensorshape badshape = ilist;
	nnet::tensorshape parts = plist;
	std::vector<double> minmax = get_double(2, "min-max", {-24, 26});
	double min = *std::min_element(minmax.begin(), minmax.end());
	double max = *std::max_element(minmax.begin(), minmax.end());

	std::shared_ptr<nnet::r_uniform_init> init = std::make_shared<nnet::r_uniform_init>();
	init->set<double>(min, max);

	nnet::variable full(shape, init, label1);
	nnet::variable full2(shape, init, label2);
	nnet::variable part(parts, init, label3);
	nnet::variable other(shape, init, label4);
	nnet::variable bad(badshape, init, label5);

	mock_observer parent({&full});
	mock_observer parent2({&full2});

	nnet::tensor* ften = full.get_tensor();
	nnet::tensor* pten = full.get_tensor();
	nnet::tensor* oten = full.get_tensor();

	EXPECT_FALSE(full.assign(&full)) <<
		"successfully assigned non-initialized variable";
	EXPECT_FALSE(full.assign(&part)) <<
		"successfully assigned non-initialized variable";
	EXPECT_FALSE(full.assign(&other)) <<
		"successfully assigned non-initialized variable";
	EXPECT_FALSE(full.assign(&bad)) <<
		"successfully assigned non-initialized variable";
	EXPECT_FALSE(part.assign(&full)) <<
		"successfully assigned non-initialized variable";
	EXPECT_FALSE(part.assign(&part)) <<
		"successfully assigned non-initialized variable";
	EXPECT_FALSE(part.assign(&other)) <<
		"successfully assigned non-initialized variable";
	EXPECT_FALSE(part.assign(&bad)) <<
		"successfully assigned non-initialized variable";

	other.initialize();
	bad.initialize();

	EXPECT_FALSE(full.assign(&bad)) <<
		sprintf("successfully var %p assigned bad variable %p", &shape, &badshape);
	EXPECT_FALSE(part.assign(&bad)) <<
		sprintf("successfully var %p assigned bad variable %p", &parts, &badshape);
	EXPECT_FALSE(ften->has_data()) <<
		"tensor ften has data";
	EXPECT_FALSE(pten->has_data()) <<
		"tensor pten has data";

	EXPECT_TRUE(full.assign(&other, true)) <<
		sprintf("failed to assigned variable %p with notification", &shape);
	EXPECT_TRUE(full2.assign(&other, false)) <<
		sprintf("failed to assigned variable %p without notification", &shape);
	EXPECT_TRUE(part.assign(&other)) <<
		sprintf("failed to assigned partial %p", &parts);
	EXPECT_TRUE(ften->has_data()) <<
		"tensor ften does not have data";
	EXPECT_TRUE(pten->has_data()) <<
		"tensor pten does not have data";

	size_t updates = testify::mocker::get_usage(&parent, "update2");
	size_t updates2 = testify::mocker::get_usage(&parent2, "update2");
	optional<std::string> updateval = testify::mocker::get_value(&parent, "update2");
	EXPECT_EQ(1, updates);
	EXPECT_EQ(0, updates2);
	ASSERT_TRUE((bool) updateval) <<
		"parent update2 value is not found";;
	EXPECT_STREQ("UPDATE", updateval->c_str());

	EXPECT_FALSE(full.assign(&full)) <<
		"successfully assigned itself";
	EXPECT_FALSE(part.assign(&part)) <<
		"successfully assigned itself";

	std::vector<double> odata = nnet::expose<double>(oten);
	std::vector<double> fdata = nnet::expose<double>(ften);
	std::vector<double> pdata = nnet::expose<double>(pten);
	EXPECT_TRUE(std::equal(odata.begin(), odata.end(), fdata.begin())) <<
		"other tensor data is not equivalent of assigned full tensor data";
	EXPECT_TRUE(std::equal(odata.begin(), odata.end(), pdata.begin())) <<
		"other tensor data is not equivalent of assigned partial tensor data";
}


#endif /* DISABLE_VARIABLE_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
