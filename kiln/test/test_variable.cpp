#ifndef DISABLE_KILN_MODULE_TESTS

#include "gtest/gtest.h"

#include "testify/mocker/mocker.hpp"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "mold/iobserver.hpp"
#include "mold/error.hpp"

#include "slip/error.hpp"

#include "kiln/variable.hpp"
#include "kiln/constant.hpp"
#include "kiln/const_init.hpp"


#ifndef DISABLE_VARIABLE_TEST


using namespace testutil;


class VARIABLE : public fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		fuzz_test::TearDown();
		kiln::Graph& g = kiln::Graph::get_global();
		assert(0 == g.size());
		testify::mocker::clear();
	}
};


struct mock_observer : public mold::iObserver, public testify::mocker
{
	mock_observer (mold::iNode* arg) :
		mold::iObserver({arg}) {}

	void initialize (void) override
	{
		label_incr("initialize");
	}

	void update (void) override
	{
		label_incr("update");
	}
};


TEST_F(VARIABLE, Init_E000)
{
	kiln::Graph& graph = kiln::Graph::get_global();
	EXPECT_EQ(0, graph.n_uninit());

	double scalar = get_double(1, "scalar")[0];
	clay::Shape shape = random_def_shape(this);
	clay::BuildTensorF builder = kiln::const_init(scalar, shape);
	std::string label = get_string(16, "label");
	kiln::Variable var(builder, label);
	mock_observer* obs = new mock_observer(var.get());

	EXPECT_EQ(0, testify::mocker::get_usage(obs, "initialize"));
	EXPECT_EQ(0, testify::mocker::get_usage(obs, "update"));
	EXPECT_EQ(1, graph.n_uninit());
	graph.initialize(var.get_uid());
	ASSERT_TRUE(var.has_data());
	clay::State state = var.get_state();
	ASSERT_SHAPEQ(shape, state.shape_);
	ASSERT_EQ(clay::DOUBLE, state.dtype_);
	double* ptr = (double*) state.data_.lock().get();
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(scalar, ptr[i]);
	}

	EXPECT_EQ(0, graph.n_uninit());
	EXPECT_EQ(1, testify::mocker::get_usage(obs, "initialize"));
	EXPECT_EQ(0, testify::mocker::get_usage(obs, "update"));
	delete obs;
}


TEST_F(VARIABLE, Assign_E001)
{
	kiln::Graph& graph = kiln::Graph::get_global();
	EXPECT_EQ(0, graph.n_uninit());

	double scalar = get_double(1, "scalar")[0];
	std::vector<size_t> clist = random_def_shape(this);
	clay::Shape shape = clist;
	clist[0]++;
	clay::Shape badshape = clist;
	std::vector<double> data = get_double(shape.n_elems(), "data");
	std::vector<double> baddata = get_double(badshape.n_elems(), "baddata");
	auto temp = get_int(shape.n_elems(), "baddata2");
	std::vector<uint16_t> baddata2(temp.begin(), temp.end());
	clay::BuildTensorF builder = kiln::const_init(scalar, shape);
	std::string label = get_string(16, "label");
	kiln::Variable var(builder, label);
	kiln::Variable var2(builder, label);
	mock_observer* obs = new mock_observer(var.get());

	kiln::Constant* c = kiln::Constant::get(data, shape);
	kiln::Constant* c2 = kiln::Constant::get(baddata, badshape);
	kiln::Constant* c3 = kiln::Constant::get(baddata2, shape);

	EXPECT_THROW(var.assign(*c), mold::UninitializedError); // var is uninit

	EXPECT_EQ(0, testify::mocker::get_usage(obs, "initialize"));
	EXPECT_EQ(0, testify::mocker::get_usage(obs, "update"));
	EXPECT_EQ(2, graph.n_uninit());
	graph.initialize(var.get_uid());
	ASSERT_TRUE(var.has_data());
	clay::State state = var.get_state();
	ASSERT_SHAPEQ(shape, state.shape_);
	ASSERT_EQ(clay::DOUBLE, state.dtype_);
	double* ptr = (double*) state.data_.lock().get();
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(scalar, ptr[i]);
	}

	EXPECT_EQ(1, graph.n_uninit());
	EXPECT_EQ(1, testify::mocker::get_usage(obs, "initialize"));
	EXPECT_EQ(0, testify::mocker::get_usage(obs, "update"));

	EXPECT_THROW(var.assign(var2), mold::UninitializedError); // var2 is uninit
	EXPECT_THROW(var.assign(*c2), slip::ShapeMismatchError); // shape mismatch
	EXPECT_THROW(var.assign(*c3), slip::TypeMismatchError); // type mismatch

	var.assign(*c);
	clay::State state2 = var.get_state();
	ASSERT_SHAPEQ(shape, state.shape_);
	ASSERT_EQ(clay::DOUBLE, state.dtype_);
	double* ptr2 = (double*) state.data_.lock().get();
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(data[i], ptr2[i]) << "failed at " << i;
	}
	EXPECT_EQ(1, graph.n_uninit());
	EXPECT_EQ(1, testify::mocker::get_usage(obs, "initialize"));
	EXPECT_EQ(1, testify::mocker::get_usage(obs, "update"));

	delete obs;
	delete c;
	delete c2;
	delete c3;
}


#endif /* DISABLE_VARIABLE_TEST */


#endif /* DISABLE_KILN_MODULE_TESTS */
