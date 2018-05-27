#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "testify/mocker/mocker.hpp"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "wire/functor.hpp"


#ifndef DISABLE_FUNCTOR_TEST


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


struct mock_operateio final : public mold::iOperateIO
{
	mock_operateio (
		std::function<void(clay::State&,std::vector<clay::State>)> op,
		std::function<clay::Shape(std::vector<clay::Shape>)> shaper,
		std::function<clay::DTYPE(std::vector<clay::DTYPE>)> typer);

	bool read_data (clay::State& dest) const override
	{
		auto outs = get_imms();
		bool success = dest.shape_.is_compatible_with(outs.first) &&
			dest.dtype_ == outs.second;
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

	ArgsF op_;

	ShaperF shaper_;

	TyperF typer_;
};


mold::Functor* junk_functor (std::vector<mold::iNode*> args,
	testify::fuzz_test* fuzzer,
	mold::GradF backward =
	[](mold::iNode* wrt, std::vector<mold::iNode*> args) -> mold::iNode*
	{
		return nullptr;
	})
{
	mold::iOperatePtrT op = mold::iOperatePtrT(new mock_operateio(
	[fuzzer](clay::State& state,std::vector<clay::State>)
	{
		size_t nbytes = state.shape_.n_elems() *
			clay::type_size(state.dtype_);
		std::string raw = fuzzer->get_string(nbytes, "mock_src_uuid");
		std::memcpy((void*) state.data_.lock().get(), raw.c_str(), nbytes);
	},
	[fuzzer](std::vector<clay::Shape>) -> clay::Shape
	{
		return random_def_shape(fuzzer, {1, 6});
	},
	[fuzzer](std::vector<clay::DTYPE>) -> clay::DTYPE
	{
		return (clay::DTYPE) fuzzer->get_int(1, "dtype",
			{1, clay::DTYPE::_SENTINEL - 1})[0];
	}));
	return new mold::Functor(args, std::move(op), backward);
}


TEST_F(FUNCTOR, Derive_G00x)
{
	mold::Variable derout;
	mold::Variable arg;
	mold::Variable arg2;
	mock_builder builder(this);
	mold::Functor* f = junk_functor(std::vector<mold::iNode*>{&arg, &arg2}, this,
	[&](mold::iNode* wrt, std::vector<mold::iNode*> args) -> mold::iNode*
	{
		EXPECT_EQ(&arg, wrt);
		EXPECT_EQ(2, args.size());
		EXPECT_EQ(&arg, args[0]);
		EXPECT_EQ(&arg2, args[1]);
		return &derout;
	});

	EXPECT_THROW(f->derive(f), std::exception);
	EXPECT_THROW(f->derive(&arg), std::exception);
	arg.initialize(builder);
	arg2.initialize(builder);
	f->initialize();
	ASSERT_TRUE(f->has_data());

	clay::State fwdstate = f->get_state();
	clay::Shape scalars(std::vector<size_t>{1});
	mold::iNode* wun = f->derive(f);
	clay::State state = wun->get_state();
	EXPECT_SHAPEQ(scalars, state.shape_);
	EXPECT_EQ(fwdstate.dtype_, state.dtype_);
	switch (fwdstate.dtype_)
	{
		case clay::DTYPE::DOUBLE:
		{
			double scalar = *((double*) state.data_.lock().get());
			EXPECT_EQ(1, scalar);
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float scalar = *((float*) state.data_.lock().get());
			EXPECT_EQ(1, scalar);
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t scalar = *((int8_t*) state.data_.lock().get());
			EXPECT_EQ(1, scalar);
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t scalar = *((uint8_t*) state.data_.lock().get());
			EXPECT_EQ(1, scalar);
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t scalar = *((int16_t*) state.data_.lock().get());
			EXPECT_EQ(1, scalar);
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t scalar = *((uint16_t*) state.data_.lock().get());
			EXPECT_EQ(1, scalar);
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t scalar = *((int32_t*) state.data_.lock().get());
			EXPECT_EQ(1, scalar);
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t scalar = *((uint32_t*) state.data_.lock().get());
			EXPECT_EQ(1, scalar);
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t scalar = *((int64_t*) state.data_.lock().get());
			EXPECT_EQ(1, scalar);
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t scalar = *((uint64_t*) state.data_.lock().get());
			EXPECT_EQ(1, scalar);
		}
		default:
		break;
	};

	mold::iNode* der = f->derive(&arg);
	EXPECT_EQ(&derout, der);

	delete f;
	delete wun;
}


#endif /* DISABLE_FUNCTOR_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */
