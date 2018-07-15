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

	clay::Shape get_shape (void) const override
	{
		return clay::Shape();
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


struct mock_operateio final : public mold::iOperateIO
{
	mock_operateio (
		std::function<void(clay::State&,std::vector<clay::State>)> op,
		std::function<clay::Shape(std::vector<clay::Shape>)> shaper,
		std::function<clay::DTYPE(std::vector<clay::DTYPE>)> typer) :
		op_(op), shaper_(shaper), typer_(typer) {}

	bool validate_data (clay::State state,
		std::vector<mold::StateRange> args) const override
	{
		std::vector<clay::Shape> shapes;
		std::vector<clay::DTYPE> types;
		for (const mold::StateRange& arg : args)
		{
			shapes.push_back(arg.shape());
			types.push_back(arg.type());
		}
		return state.shape_.is_compatible_with(shaper_(shapes)) &&
			state.dtype_ == typer_(types);
	}

	bool write_data (clay::State& dest,
		std::vector<mold::StateRange> args) const override
	{
		std::vector<clay::Shape> shapes;
		std::vector<clay::DTYPE> types;
		std::vector<clay::State> states;
		for (const mold::StateRange& arg : args)
		{
			shapes.push_back(arg.shape());
			types.push_back(arg.type());
			states.push_back(arg.arg_);
		}
		clay::Shape oshape = shaper_(shapes);
		clay::DTYPE otype = typer_(types);
		bool success = dest.shape_.is_compatible_with(oshape) &&
			dest.dtype_ == otype;
		if (success)
		{
			op_(dest, states);
		}
		return success;
	}

	clay::TensorPtrT make_data (std::vector<mold::StateRange> args) const override
	{
		std::vector<clay::Shape> shapes;
		std::vector<clay::DTYPE> types;
		for (const mold::StateRange& arg : args)
		{
			shapes.push_back(arg.shape());
			types.push_back(arg.type());
		}
		clay::Tensor* out = new clay::Tensor(shaper_(shapes), typer_(types));
		clay::State dest = out->get_state();
		write_data(dest, args);
		return clay::TensorPtrT(out);
	}

private:
	iOperateIO* clone_impl (void) const override
	{
		return new mock_operateio(*this);
	}

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
	mold::OperatePtrT op = mold::OperatePtrT(new mock_operateio(
	[fuzzer](clay::State& state,std::vector<clay::State>)
	{
		size_t nbytes = state.shape_.n_elems() *
			clay::type_size(state.dtype_);
		std::string raw = fuzzer->get_string(nbytes, "mock_src_uuid");
		std::memcpy((void*) state.get(), raw.c_str(), nbytes);
	},
	[shape](std::vector<clay::Shape>) -> clay::Shape
	{
		return shape;
	},
	[dtype](std::vector<clay::DTYPE>) -> clay::DTYPE
	{
		return dtype;
	}));
	std::vector<mold::NodeRange> dims;
	std::transform(args.begin(), args.end(), std::back_inserter(dims),
	[](mold::iNode* node)
	{
		return mold::NodeRange{node,mold::Range{0,0}};
	});
	return new mold::Functor(dims, std::move(op));
}


std::string fake_init (clay::Tensor* tens, testify::fuzz_test* fuzzer)
{
	clay::State state = tens->get_state();
	size_t nbytes = state.shape_.n_elems() * clay::type_size(state.dtype_);
	std::string uuid = fuzzer->get_string(nbytes, "uuid");
	std::memcpy(state.get(), uuid.c_str(), nbytes);
	return uuid;
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
	mold::iNode* clone = f->clone();
	*assign = *f;

	auto aud = arg.get_audience();
	auto aud2 = arg2.get_audience();
	auto nonaud = non.get_audience();
	EXPECT_EQ(0, nonaud.size());
	EXPECT_TRUE(aud.end() != aud.find(cp)) << "cp not found in audience of arg";
	EXPECT_TRUE(aud.end() != aud.find(
		dynamic_cast<mold::Functor*>(clone))) << "clone not found in audience of arg";
	EXPECT_TRUE(aud.end() != aud.find(assign)) << "assign not found in audience of arg";
	EXPECT_TRUE(aud2.end() != aud2.find(cp)) << "cp not found in audience of arg2";
	EXPECT_TRUE(aud2.end() != aud2.find(
		dynamic_cast<mold::Functor*>(clone))) << "clone not found in audience of arg2";
	EXPECT_TRUE(aud2.end() != aud2.find(assign)) << "assign not found in audience of arg2";

	delete f;
	delete cp;
	delete clone;
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
	clay::Shape shape = random_def_shape(this, {2, 6});
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype",
		{1, clay::DTYPE::_SENTINEL - 1})[0];
	mold::Functor* f = junk_functor(std::vector<mold::iNode*>{&arg}, this);

	EXPECT_FALSE(f->has_data());
	arg.set_data(clay::TensorPtrT(new clay::Tensor(shape, dtype)));
	f->initialize();
	EXPECT_TRUE(f->has_data());
	delete f;
}


TEST_F(FUNCTOR, GetState_D004)
{
	mold::OperatePtrT identity = mold::OperatePtrT(new mock_operateio(
	[](clay::State& out,std::vector<clay::State> in)
	{
		size_t nbytes = out.shape_.n_elems() *
			clay::type_size(out.dtype_);
		std::memcpy(out.get(),
			in[0].get(), nbytes);
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
	clay::Shape shape = random_def_shape(this, {2, 6});
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype",
		{1, clay::DTYPE::_SENTINEL - 1})[0];
	mold::Functor* f = new mold::Functor(
		std::vector<mold::NodeRange>{{&arg, {0, 0}}},
		std::move(identity));

	EXPECT_THROW(f->get_state(), mold::UninitializedError);
	clay::Tensor* tens = new clay::Tensor(shape, dtype);
	std::string uuid = fake_init(tens, this);
	arg.set_data(clay::TensorPtrT(tens));
	f->initialize();
	clay::State state = f->get_state();
	ASSERT_SHAPEQ(shape, state.shape_);
	clay::Shape fshape = f->get_shape();
	EXPECT_SHAPEQ(shape, fshape);
	ASSERT_EQ(dtype, state.dtype_);
	std::string got_uuid(state.get(),
		state.shape_.n_elems() * clay::type_size(state.dtype_));
	EXPECT_STREQ(uuid.c_str(), got_uuid.c_str());

	delete f;
}


#endif /* DISABLE_FUNCTOR_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
