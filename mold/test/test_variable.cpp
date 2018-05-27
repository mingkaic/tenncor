#ifndef DISABLE_MOLD_MODULE_TESTS

#include "gtest/gtest.h"

#include "testify/mocker/mocker.hpp"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "ioutil/stream.hpp"

#include "mold/sink.hpp"
#include "mold/variable.hpp"


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


struct mock_source final : public clay::iSource, public testify::mocker
{
	mock_source (clay::Shape shape, clay::DTYPE dtype, testify::fuzz_test* fuzzer)
	{
		size_t nbytes = shape.n_elems() * clay::type_size(dtype);
		uuid_ = fuzzer->get_string(nbytes, "mock_src_uuid");

		ptr_ = clay::make_char(nbytes);
		std::memcpy(ptr_.get(), uuid_.c_str(), nbytes);

		state_ = {ptr_, shape, dtype};
	}

	bool read_data (clay::State& dest) const override
	{
		bool success = false == uuid_.empty() &&
			dest.dtype_ == state_.dtype_ &&
			dest.shape_.is_compatible_with(state_.shape_);
		if (success)
		{
			std::memcpy((void*) dest.data_.lock().get(), ptr_.get(), uuid_.size());
			label_incr("read_data_success");
		}
		label_incr("read_data");
		return success;
	}

	clay::State state_;

	std::shared_ptr<char> ptr_;

	std::string uuid_;
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


struct mock_builder final : public clay::iBuilder, public testify::mocker
{
	mock_builder (testify::fuzz_test* fuzzer) :
		shape_(random_def_shape(fuzzer, {2, 6})),
		dtype_((clay::DTYPE) fuzzer->get_int(1, "dtype",
		{1, clay::DTYPE::_SENTINEL - 1})[0])
	{
		size_t nbytes = shape_.n_elems() * clay::type_size(dtype_);
		uuid_ = fuzzer->get_string(nbytes, "uuid_");
		ptr_ = clay::make_char(nbytes);
		std::memcpy(ptr_.get(), uuid_.c_str(), nbytes);
	}

	clay::TensorPtrT get (void) const override
	{
		label_incr("get");
		return clay::TensorPtrT(new clay::Tensor(ptr_, shape_, dtype_));
	}

	clay::TensorPtrT get (clay::Shape shape) const override
	{
		label_incr("getwshape");
		ioutil::Stream str;
		str << shape.as_list();
		set_label("getwshape", str.str());
		return clay::TensorPtrT(new clay::Tensor(ptr_, shape_, dtype_));
	}

	clay::Shape shape_;
	clay::DTYPE dtype_;
	std::string uuid_;
	std::shared_ptr<char> ptr_;

protected:
	clay::iBuilder* clone_impl (void) const override
	{
		return new mock_builder(*this);
	}
};


TEST_F(VARIABLE, Copy_C000)
{
	mock_builder builder(this);
	mold::Variable assign;
	mold::Variable assign1;
	mold::Variable assign2;
	assign.initialize(builder);
	assign2.initialize(builder);

	mold::Variable var;
	mold::Variable var2;

	mock_builder builder2(this);
	var.initialize(builder2);

	mold::Variable cp(var);
	mold::Variable cp2(var2);
	ASSERT_TRUE(cp.has_data());
	clay::State state = cp.get_state();
	std::string got_uuid(state.data_.lock().get(),
		state.shape_.n_elems() * clay::type_size(state.dtype_));
	EXPECT_STREQ(builder2.uuid_.c_str(), got_uuid.c_str());
	EXPECT_SHAPEQ(builder2.shape_, state.shape_);
	EXPECT_EQ(builder2.dtype_, state.dtype_);

	EXPECT_FALSE(cp2.has_data());

	assign = var;
	assign1 = var;
	assign2 = var2;
	ASSERT_TRUE(assign.has_data());
	ASSERT_TRUE(assign1.has_data());
	clay::State state2 = assign.get_state();
	std::string got_uuid2(state2.data_.lock().get(),
		state2.shape_.n_elems() * clay::type_size(state2.dtype_));
	EXPECT_STREQ(builder2.uuid_.c_str(), got_uuid2.c_str());
	EXPECT_SHAPEQ(builder2.shape_, state2.shape_);
	EXPECT_EQ(builder2.dtype_, state2.dtype_);
	clay::State state3 = assign1.get_state();
	std::string got_uuid3(state3.data_.lock().get(),
		state3.shape_.n_elems() * clay::type_size(state3.dtype_));
	EXPECT_STREQ(builder2.uuid_.c_str(), got_uuid3.c_str());
	EXPECT_SHAPEQ(builder2.shape_, state3.shape_);
	EXPECT_EQ(builder2.dtype_, state3.dtype_);

	EXPECT_FALSE(assign2.has_data());
}


TEST_F(VARIABLE, Move_C001)
{
	mock_builder builder(this);
	mold::Variable assign;
	mold::Variable assign2;
	assign.initialize(builder);
	assign2.initialize(builder);

	mold::Variable var;
	mold::Variable var2;

	mock_builder builder2(this);
	var.initialize(builder2);

	mold::Variable cp(std::move(var));
	mold::Variable cp2(std::move(var2));
	ASSERT_TRUE(cp.has_data());
	clay::State state = cp.get_state();
	std::string got_uuid(state.data_.lock().get(),
		state.shape_.n_elems() * clay::type_size(state.dtype_));
	EXPECT_STREQ(builder2.uuid_.c_str(), got_uuid.c_str());
	EXPECT_SHAPEQ(builder2.shape_, state.shape_);
	EXPECT_EQ(builder2.dtype_, state.dtype_);

	EXPECT_FALSE(cp2.has_data());
	EXPECT_FALSE(var.has_data());
	EXPECT_FALSE(var2.has_data());

	assign = std::move(cp);
	assign2 = std::move(cp2);
	ASSERT_TRUE(assign.has_data());
	clay::State state2 = assign.get_state();
	std::string got_uuid2(state2.data_.lock().get(),
		state2.shape_.n_elems() * clay::type_size(state2.dtype_));
	EXPECT_STREQ(builder2.uuid_.c_str(), got_uuid2.c_str());
	EXPECT_SHAPEQ(builder2.shape_, state2.shape_);
	EXPECT_EQ(builder2.dtype_, state2.dtype_);

	EXPECT_FALSE(assign2.has_data());
	EXPECT_FALSE(cp.has_data());
	EXPECT_FALSE(cp2.has_data());
}


TEST_F(VARIABLE, Data_C002)
{
	mold::Variable var;
	mold::Variable var2;
	mock_observer* obs = new mock_observer(&var);
	mock_observer* obs2 = new mock_observer(&var2);
	mock_builder builder(this);
	clay::Shape shape = random_def_shape(this);

	EXPECT_FALSE(var.has_data()) << "uninitialized variable has data";
	EXPECT_EQ(0, testify::mocker::get_usage(&builder, "get"));
	var.initialize(builder);
	EXPECT_EQ(1, testify::mocker::get_usage(obs, "initialize"));
	EXPECT_EQ(1, testify::mocker::get_usage(&builder, "get"));
	EXPECT_TRUE(var.has_data()) << "initialized variable doesn't have data";

	EXPECT_FALSE(var2.has_data()) << "uninitialized variable has data";
	EXPECT_EQ(0, testify::mocker::get_usage(&builder, "getwshape"));
	var2.initialize(builder, shape);
	EXPECT_EQ(1, testify::mocker::get_usage(obs2, "initialize"));
	EXPECT_EQ(1, testify::mocker::get_usage(&builder, "getwshape"));
	EXPECT_TRUE(var2.has_data()) << "initialized variable doesn't have data";
	optional<std::string> initshape = testify::mocker::get_value(&builder, "getwshape");

	delete obs;
	delete obs2;
}


TEST_F(VARIABLE, State_C003)
{
	mold::Variable var;
	mock_observer* obs = new mock_observer(&var);
	mock_builder builder(this);

	EXPECT_THROW(var.get_state(), std::exception);
	EXPECT_EQ(0, testify::mocker::get_usage(&builder, "get"));
	var.initialize(builder);
	EXPECT_EQ(1, testify::mocker::get_usage(&builder, "get"));
	EXPECT_EQ(1, testify::mocker::get_usage(obs, "initialize"));
	clay::State state = var.get_state();
	std::string got_uuid(state.data_.lock().get(),
		state.shape_.n_elems() * clay::type_size(state.dtype_));
	EXPECT_STREQ(builder.uuid_.c_str(), got_uuid.c_str());
	EXPECT_SHAPEQ(builder.shape_, state.shape_);
	EXPECT_EQ(builder.dtype_, state.dtype_);

	delete obs;
}


TEST_F(VARIABLE, Assign_C004)
{
	mold::Variable var;
	mock_observer* obs = new mock_observer(&var);
	mock_builder builder(this);
	mock_source src(builder.shape_, builder.dtype_, this);

	EXPECT_THROW(var.assign(src), std::exception);
	EXPECT_EQ(0, testify::mocker::get_usage(&builder, "get"));
	var.initialize(builder);
	EXPECT_EQ(1, testify::mocker::get_usage(&builder, "get"));
	EXPECT_EQ(1, testify::mocker::get_usage(obs, "initialize"));
	EXPECT_EQ(0, testify::mocker::get_usage(obs, "update"));
	var.assign(src);
	EXPECT_EQ(1, testify::mocker::get_usage(&src, "read_data_success"));
	EXPECT_EQ(1, testify::mocker::get_usage(&src, "read_data"));
	EXPECT_EQ(1, testify::mocker::get_usage(obs, "update"));
	var.get_state();
	clay::State state = var.get_state();
	std::string got_uuid(state.data_.lock().get(),
		state.shape_.n_elems() * clay::type_size(state.dtype_));
	EXPECT_STREQ(src.uuid_.c_str(), got_uuid.c_str());
	EXPECT_SHAPEQ(src.state_.shape_, state.shape_);
	EXPECT_EQ(src.state_.dtype_, state.dtype_);

	delete obs;
}


#endif /* DISABLE_VARIABLE_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
