#ifndef DISABLE_MOLD_MODULE_TESTS

#include "gtest/gtest.h"

#include "testify/mocker/mocker.hpp"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "ioutil/stream.hpp"

#include "clay/memory.hpp"

#include "mold/sink.hpp"
#include "mold/variable.hpp"
#include "mold/iobserver.hpp"
#include "mold/error.hpp"


#ifndef DISABLE_VARIABLE_TEST


using namespace testutil;


class VARIABLE : public fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		fuzz_test::TearDown();
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


std::string fake_init_v (clay::Tensor* tens, testify::fuzz_test* fuzzer)
{
	clay::State state = tens->get_state();
	size_t nbytes = state.shape_.n_elems() * clay::type_size(state.dtype_);
	std::string uuid = fuzzer->get_string(nbytes, "uuid");
	std::memcpy(state.get(), uuid.c_str(), nbytes);
	return uuid;
	}


TEST_F(VARIABLE, Copy_C000)
{
	clay::Shape shape = random_def_shape(this, {2, 6});
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype",
		{1, clay::DTYPE::_SENTINEL - 1})[0];
	clay::Tensor* tens = new clay::Tensor(shape, dtype);
	clay::Tensor* ten2s = new clay::Tensor(shape, dtype);
	mold::Variable assign;
	mold::Variable assign1;
	mold::Variable assign2;
	assign.initialize(clay::TensorPtrT(tens));
	assign2.initialize(clay::TensorPtrT(ten2s));

	mold::Variable var;
	mold::Variable var2;

	clay::Shape shape2 = random_def_shape(this, {2, 6});
	clay::DTYPE dtype2 = (clay::DTYPE) get_int(1, "dtype",
		{1, clay::DTYPE::_SENTINEL - 1})[0];
	clay::Tensor* tens2 = new clay::Tensor(shape2, dtype2);
	std::string uuid2 = fake_init_v(tens2, this);
	var.initialize(clay::TensorPtrT(tens2));

	mold::Variable cp(var);
	mold::Variable cp2(var2);
	ASSERT_TRUE(cp.has_data());
	clay::State state = cp.get_state();
	std::string got_uuid(state.get(),
		state.shape_.n_elems() * clay::type_size(state.dtype_));
	EXPECT_STREQ(uuid2.c_str(), got_uuid.c_str());
	EXPECT_SHAPEQ(shape2, state.shape_);
	EXPECT_EQ(dtype2, state.dtype_);

	EXPECT_FALSE(cp2.has_data());

	assign = var;
	assign1 = var;
	assign2 = var2;
	ASSERT_TRUE(assign.has_data());
	ASSERT_TRUE(assign1.has_data());
	clay::State state2 = assign.get_state();
	std::string got_uuid2(state2.get(),
		state2.shape_.n_elems() * clay::type_size(state2.dtype_));
	EXPECT_STREQ(uuid2.c_str(), got_uuid2.c_str());
	EXPECT_SHAPEQ(shape2, state2.shape_);
	EXPECT_EQ(dtype2, state2.dtype_);
	clay::State state3 = assign1.get_state();
	std::string got_uuid3(state3.get(),
		state3.shape_.n_elems() * clay::type_size(state3.dtype_));
	EXPECT_STREQ(uuid2.c_str(), got_uuid3.c_str());
	EXPECT_SHAPEQ(shape2, state3.shape_);
	EXPECT_EQ(dtype2, state3.dtype_);

	EXPECT_FALSE(assign2.has_data());
}


TEST_F(VARIABLE, Move_C001)
{
	clay::Shape shape = random_def_shape(this, {2, 6});
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype",
		{1, clay::DTYPE::_SENTINEL - 1})[0];
	clay::Tensor* tens = new clay::Tensor(shape, dtype);
	clay::Tensor* ten2s = new clay::Tensor(shape, dtype);
	mold::Variable assign;
	mold::Variable assign2;
	assign.initialize(clay::TensorPtrT(tens));
	assign2.initialize(clay::TensorPtrT(ten2s));

	mold::Variable var;
	mold::Variable var2;

	clay::Shape shape2 = random_def_shape(this, {2, 6});
	clay::DTYPE dtype2 = (clay::DTYPE) get_int(1, "dtype",
		{1, clay::DTYPE::_SENTINEL - 1})[0];
	clay::Tensor* tens2 = new clay::Tensor(shape2, dtype2);
	std::string uuid2 = fake_init_v(tens2, this);
	var.initialize(clay::TensorPtrT(tens2));

	mold::Variable cp(std::move(var));
	mold::Variable cp2(std::move(var2));
	ASSERT_TRUE(cp.has_data());
	clay::State state = cp.get_state();
	std::string got_uuid(state.get(),
		state.shape_.n_elems() * clay::type_size(state.dtype_));
	EXPECT_STREQ(uuid2.c_str(), got_uuid.c_str());
	EXPECT_SHAPEQ(shape2, state.shape_);
	EXPECT_EQ(dtype2, state.dtype_);

	EXPECT_FALSE(cp2.has_data());
	EXPECT_FALSE(var.has_data());
	EXPECT_FALSE(var2.has_data());

	assign = std::move(cp);
	assign2 = std::move(cp2);
	ASSERT_TRUE(assign.has_data());
	clay::State state2 = assign.get_state();
	std::string got_uuid2(state2.get(),
		state2.shape_.n_elems() * clay::type_size(state2.dtype_));
	EXPECT_STREQ(uuid2.c_str(), got_uuid2.c_str());
	EXPECT_SHAPEQ(shape2, state2.shape_);
	EXPECT_EQ(dtype2, state2.dtype_);

	EXPECT_FALSE(assign2.has_data());
	EXPECT_FALSE(cp.has_data());
	EXPECT_FALSE(cp2.has_data());
}


TEST_F(VARIABLE, Data_C002)
{
	mold::Variable var;
	mold::Variable var2;
	mock_observer* obs = new mock_observer(&var);
	clay::Shape shape = random_def_shape(this, {2, 6});
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype",
		{1, clay::DTYPE::_SENTINEL - 1})[0];
	clay::Tensor* tens = new clay::Tensor(shape, dtype);
	std::string uuid = fake_init_v(tens, this);

	EXPECT_FALSE(var.has_data()) << "uninitialized variable has data";
	var.initialize(clay::TensorPtrT(tens));
	EXPECT_EQ(1, testify::mocker::get_usage(obs, "initialize"));
	EXPECT_TRUE(var.has_data()) << "initialized variable doesn't have data";

	delete obs;
}


TEST_F(VARIABLE, State_C003)
{
	mold::Variable var;
	mock_observer* obs = new mock_observer(&var);
	clay::Shape shape = random_def_shape(this, {2, 6});
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype",
		{1, clay::DTYPE::_SENTINEL - 1})[0];

	EXPECT_THROW(var.get_state(), mold::UninitializedError);
	clay::Tensor* tens = new clay::Tensor(shape, dtype);
	std::string uuid = fake_init_v(tens, this);
	var.initialize(clay::TensorPtrT(tens));
	EXPECT_EQ(1, testify::mocker::get_usage(obs, "initialize"));
	clay::State state = var.get_state();
	std::string got_uuid(state.get(),
		state.shape_.n_elems() * clay::type_size(state.dtype_));
	EXPECT_STREQ(uuid.c_str(), got_uuid.c_str());
	EXPECT_SHAPEQ(shape, state.shape_);
	EXPECT_EQ(dtype, state.dtype_);

	delete obs;
}


#endif /* DISABLE_VARIABLE_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
