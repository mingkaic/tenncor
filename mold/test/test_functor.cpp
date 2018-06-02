#ifndef DISABLE_MOLD_MODULE_TESTS

#include "gtest/gtest.h"

#include "testify/mocker/mocker.hpp"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "clay/memory.hpp"

#include "mold/inode.hpp"
#include "mold/variable.hpp"
#include "mold/functor.hpp"
#include "mold/sink.hpp"
#include "mold/error.hpp"


#ifndef DISABLE_FUNCTOR_TEST


using namespace testutil;


class FUNCTOR : public fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		fuzz_test::TearDown();
		testify::mocker::clear();
	}
};


struct mock_node final : public mold::iNode, public testify::mocker
{
	bool has_data (void) const override
	{
		return true;
	}

	clay::State get_state (void) const override
	{
		return clay::State();
	}

protected:
	iNode* clone_impl (void) const override
	{
		return new mock_node(*this);
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


struct mock_operateio final : public mold::iOperateIO
{
	mock_operateio (
		std::function<void(clay::State&,std::vector<clay::State>)> op,
		std::function<clay::Shape(std::vector<clay::Shape>)> shaper,
		std::function<clay::DTYPE(std::vector<clay::DTYPE>)> typer) :
		op_(op), shaper_(shaper), typer_(typer) {}

	bool read_data (clay::State& dest) const override
	{
		std::vector<clay::Shape> shapes;
		std::vector<clay::DTYPE> types;
		for (const clay::State& arg : args_)
		{
			shapes.push_back(arg.shape_);
			types.push_back(arg.dtype_);
		}
		clay::Shape oshape = shaper_(shapes);
		clay::DTYPE otype = typer_(types);
		bool success = dest.shape_.is_compatible_with(oshape) &&
			dest.dtype_ == otype;
		if (success)
		{
			op_(dest, args_);
		}
		return success;
	}

	mold::ImmPair get_imms (void) override
	{
		std::vector<clay::Shape> shapes;
		std::vector<clay::DTYPE> types;
		for (const clay::State& arg : args_)
		{
			shapes.push_back(arg.shape_);
			types.push_back(arg.dtype_);
		}
		return {shaper_(shapes), typer_(types)};
	}

	void set_args (std::vector<clay::State> args) override
	{
		args_ = args;
	}

private:
	iOperateIO* clone_impl (void) const override
	{
		return new mock_operateio(*this);
	}

	std::vector<clay::State> args_;

	std::function<void(clay::State&,std::vector<clay::State>)> op_;

	std::function<clay::Shape(std::vector<clay::Shape>)> shaper_;

	std::function<clay::DTYPE(std::vector<clay::DTYPE>)> typer_;
};


mold::Functor* junk_functor (std::vector<mold::iNode*> args,
	testify::fuzz_test* fuzzer)
{
	clay::Shape shape = random_def_shape(fuzzer, {1, 6});
	clay::DTYPE dtype = (clay::DTYPE) fuzzer->get_int(1, "dtype",
		{1, clay::DTYPE::_SENTINEL - 1})[0];
	mold::iOperatePtrT op = mold::iOperatePtrT(new mock_operateio(
	[fuzzer](clay::State& state,std::vector<clay::State>)
	{
		size_t nbytes = state.shape_.n_elems() *
			clay::type_size(state.dtype_);
		std::string raw = fuzzer->get_string(nbytes, "mock_src_uuid");
		std::memcpy((void*) state.data_.lock().get(), raw.c_str(), nbytes);
	},
	[shape](std::vector<clay::Shape>) -> clay::Shape
	{
		return shape;
	},
	[dtype](std::vector<clay::DTYPE>) -> clay::DTYPE
	{
		return dtype;
	}));
	return new mold::Functor(args, std::move(op));
}


TEST_F(FUNCTOR, Copy_D000)
{
	mock_node non;
	mock_node arg;
	mock_node arg2;

	mold::Functor* assign = junk_functor(std::vector<mold::iNode*>{&non}, this);
	mold::Functor* f = junk_functor(std::vector<mold::iNode*>{&arg, &arg2}, this);

	ASSERT_TRUE(f->has_data());

	mold::Functor* cp = new mold::Functor(*f);
	*assign = *f;

	auto aud = arg.get_audience();
	auto aud2 = arg2.get_audience();
	auto nonaud = non.get_audience();
	EXPECT_EQ(0, nonaud.size());
	EXPECT_TRUE(aud.end() != aud.find(cp)) << "cp not found in audience of arg";
	EXPECT_TRUE(aud.end() != aud.find(assign)) << "assign not found in audience of arg";
	EXPECT_TRUE(aud2.end() != aud2.find(cp)) << "cp not found in audience of arg2";
	EXPECT_TRUE(aud2.end() != aud2.find(assign)) << "assign not found in audience of arg2";

	delete f;
	delete cp;
	delete assign;
}


TEST_F(FUNCTOR, Move_D001)
{
	mock_node non;
	mock_node arg;
	mock_node arg2;

	mold::Functor* assign = junk_functor(std::vector<mold::iNode*>{&non}, this);
	mold::Functor* f = junk_functor(std::vector<mold::iNode*>{&arg, &arg2}, this);

	ASSERT_TRUE(f->has_data());

	mold::Functor* mv = new mold::Functor(std::move(*f));
	auto aud = arg.get_audience();
	auto aud2 = arg2.get_audience();
	EXPECT_TRUE(aud.end() != aud.find(mv)) << "mv not found in audience of arg";
	EXPECT_TRUE(aud2.end() != aud2.find(mv)) << "mv not found in audience of arg2";
	EXPECT_TRUE(aud.end() == aud.find(f)) << "moved f found in audience of arg";
	EXPECT_TRUE(aud2.end() == aud2.find(f)) << "moved f found in audience of arg2";

	*assign = std::move(*mv);
	aud = arg.get_audience();
	aud2 = arg2.get_audience();
	auto nonaud = non.get_audience();
	EXPECT_EQ(0, nonaud.size());
	EXPECT_TRUE(aud.end() != aud.find(assign)) << "assign not found in audience of arg";
	EXPECT_TRUE(aud2.end() != aud2.find(assign)) << "assign not found in audience of arg2";
	EXPECT_TRUE(aud.end() == aud.find(f)) << "moved mv found in audience of arg";
	EXPECT_TRUE(aud2.end() == aud2.find(f)) << "moved mv found in audience of arg2";

	delete f;
	delete mv;
	delete assign;
}


TEST_F(FUNCTOR, Death_D002)
{
	mock_node* arg = new mock_node();
	mock_node arg2;
	mold::Sink s(junk_functor(std::vector<mold::iNode*>{arg, &arg2}, this));

	delete arg;
	EXPECT_TRUE(s.expired());
}


TEST_F(FUNCTOR, HasData_D003)
{
	mold::Variable arg;
	mock_builder builder(this);
	mold::Functor* f = junk_functor(std::vector<mold::iNode*>{&arg}, this);

	EXPECT_FALSE(f->has_data());
	arg.initialize(builder);

	EXPECT_TRUE(f->has_data());
	delete f;
}


TEST_F(FUNCTOR, GetState_D004)
{
	mold::iOperatePtrT identity = mold::iOperatePtrT(new mock_operateio(
	[](clay::State& out,std::vector<clay::State> in)
	{
		size_t nbytes = out.shape_.n_elems() *
			clay::type_size(out.dtype_);
		std::memcpy((void*) out.data_.lock().get(),
			in[0].data_.lock().get(), nbytes);
	},
	[](std::vector<clay::Shape> in) -> clay::Shape
	{
		return in[0];
	},
	[](std::vector<clay::DTYPE> in) -> clay::DTYPE
	{
		return in[0];
	}));
	mold::Variable arg;
	mock_builder builder(this);
	mock_source src(builder.shape_, builder.dtype_, this);
	mold::Functor* f = new mold::Functor(std::vector<mold::iNode*>{&arg},
		std::move(identity));

	EXPECT_THROW(f->get_state(), mold::UninitializedError);
	arg.initialize(builder);
	clay::State state = f->get_state();
	std::string got_uuid(state.data_.lock().get(),
		state.shape_.n_elems() * clay::type_size(state.dtype_));
	EXPECT_STREQ(builder.uuid_.c_str(), got_uuid.c_str());
	EXPECT_SHAPEQ(builder.shape_, state.shape_);
	EXPECT_EQ(builder.dtype_, state.dtype_);

	arg.assign(src);
	clay::State state2 = f->get_state();
	std::string got_uuid2(state2.data_.lock().get(),
		state2.shape_.n_elems() * clay::type_size(state2.dtype_));
	EXPECT_STREQ(src.uuid_.c_str(), got_uuid2.c_str());
	EXPECT_SHAPEQ(src.state_.shape_, state2.shape_);
	EXPECT_EQ(src.state_.dtype_, state2.dtype_);

	delete f;
}


#endif /* DISABLE_FUNCTOR_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
