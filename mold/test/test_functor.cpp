#ifndef DISABLE_MOLD_MODULE_TESTS

#include "gtest/gtest.h"

#include "testify/mocker/mocker.hpp" 

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "mold/inode.hpp"
#include "mold/variable.hpp"
#include "mold/functor.hpp"
#include "mold/sink.hpp"


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

	mold::iNode* derive (mold::iNode* wrt) override
	{
		return nullptr;
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


mold::Functor* junk_functor (std::vector<mold::iNode*> args,
	testify::fuzz_test* fuzzer,
	mold::GradF backward = 
	[](mold::iNode* wrt, std::vector<mold::iNode*> args) -> mold::iNode*
	{
		return nullptr;
	})
{
	mold::OperateIO op(
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
	});
	return new mold::Functor(args, op, backward);
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
	mold::OperateIO identity(
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
	});
	mold::Variable arg;
	mock_builder builder(this);
	mock_source src(builder.shape_, builder.dtype_, this);
	mold::Functor* f = new mold::Functor(std::vector<mold::iNode*>{&arg}, identity,
	[](mold::iNode* wrt, std::vector<mold::iNode*> args) -> mold::iNode*
	{
		return nullptr;
	});

	EXPECT_THROW(f->get_state(), std::exception);
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


TEST_F(FUNCTOR, Derive_D005)
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


TEST_F(FUNCTOR, Prop_D006)
{
	
}


#endif /* DISABLE_FUNCTOR_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
