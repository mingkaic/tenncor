#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "clay/memory.hpp"

#include "wire/operators.hpp"
#include "wire/variable.hpp"


#ifndef DISABLE_OPERATORS_TEST


using namespace testutil;


class OPERATORS : public fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		testutil::fuzz_test::TearDown();
		wire::Graph& g = wire::Graph::get_global();
		assert(0 == g.size());
	}
};


struct mock_builder final : public clay::iBuilder
{
	mock_builder (testify::fuzz_test* fuzzer) :
		mock_builder(fuzzer,
		(clay::DTYPE) fuzzer->get_int(1, "dtype",
		{1, clay::DTYPE::_SENTINEL - 1})[0]) {}

	mock_builder (testify::fuzz_test* fuzzer, clay::DTYPE dtype) :
		shape_(random_def_shape(fuzzer, {2, 6})), dtype_(dtype)
	{
		size_t nbytes = shape_.n_elems() * clay::type_size(dtype_);
		uuid_ = fuzzer->get_string(nbytes, "uuid_");
		ptr_ = clay::make_char(nbytes);
		std::memcpy(ptr_.get(), uuid_.c_str(), nbytes);
	}

	clay::TensorPtrT get (void) const override
	{
		return clay::TensorPtrT(new clay::Tensor(ptr_, shape_, dtype_));
	}

	clay::TensorPtrT get (clay::Shape shape) const override
	{
		return clay::TensorPtrT(new clay::Tensor(ptr_, shape, dtype_));
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


using VARFUNC = std::function<wire::Identifier*(std::vector<wire::Identifier*>)>;


using SCALAR = std::function<double(double)>;


using GRADER = std::function<std::vector<double>(std::vector<double> input)>;


static void selfGrad (wire::Identifier* f, clay::State state)
{
	clay::Shape scalars(std::vector<size_t>{1});
	wire::Identifier* wun = f->derive(f);
	clay::State back = wun->get_state();
	EXPECT_SHAPEQ(scalars, back.shape_);
	EXPECT_EQ(state.dtype_, back.dtype_);
	switch (state.dtype_)
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
}


// static void unarElemTest (fuzz_test* fuzzer, slip::OPCODE opcode, VARFUNC op,
// 	SCALAR expect, GRADER grad, std::pair<double,double> limits = {-1, 1})
// {
// 	mock_builder builder(this, clay::DOUBLE);
// 	wire::Variable arg(builder, arglabel);

// 	// test behavior G000
// 	wire::Identifier* f = op({&leaf});
// 	wire::Identifier* f2 = op({&leaf});
// 	EXPECT_EQ(f, f2);

// 	// test behavior G001
// 	EXPECT_EQ(nullptr, op({nullptr}));

// 	// test behavior G002
// 	EXPECT_THROW(f->derive(f), std::exception);
// 	EXPECT_THROW(f->derive(&arg), std::exception);

// 	wire::Graph::get_global().initialize_all();
// 	ASSERT_TRUE(f->has_data());
// 	clay::State state = f->get_state();
// 	// test behavior G003
// 	selfGrad(f, state);

// 	// test behavior B1xx
// 	wire::Identifier* der = f->derive(&arg);
// 	EXPECT_SHAPEQ(builder.shape_, state.shape_);
// 	double* inptr = (double*) builder.data_.get();
// 	double* outptr = (double*) state.data_.get();
// 	std::vector<double> input(inptr, inptr + builder.shape_.n_elems());
// 	std::vector<double> output(outptr, outptr + state.shape_.n_elems());
// 	for (size_t i = 0; i < n; ++i)
// 	{
// 		EXPECT_EQ(expect(input[i]), output[i]);
// 	}

// 	clay::State back = der->get_state();
// 	double* backptr = (double*) back.data_.get();
// 	std::vector<double> backput(backptr, backptr + back.shape_.n_elems());
// 	std::vector<double> expect_grad = grad(input);
// 	EXPECT_ARREQ(backput, expect_grad);
// }


TEST_F(OPERATORS, Derive_G00x)
{
	// slip::OPCODE opcode = (slip::OPCODE) get_int(1, "opcode", {0, slip::_SENTINEL - 1})[0];

	std::string derlabel = get_string(16, "derlabel");
	std::string arglabel = get_string(16, "arglabel");
	std::string arg2label = get_string(16, "arg2label");
	mock_builder builder(this);
	wire::Variable derout(builder, derlabel);
	wire::Variable arg(builder, arglabel);
	wire::Variable arg2(builder, arg2label);
	wire::Functor* f = new wire::Functor(std::vector<wire::Identifier*>{&arg, &arg2}, slip::ADD,
	[&](wire::Identifier* wrt, std::vector<wire::Identifier*> args) -> wire::Identifier*
	{
		EXPECT_EQ(&arg, wrt);
		EXPECT_EQ(2, args.size());
		EXPECT_EQ(&arg, args[0]);
		EXPECT_EQ(&arg2, args[1]);
		return &derout;
	});

	EXPECT_THROW(f->derive(f), std::exception);
	EXPECT_THROW(f->derive(&arg), std::exception);
	wire::Graph::get_global().initialize_all();
	ASSERT_TRUE(f->has_data());

	clay::State fwdstate = f->get_state();
	wire::Identifier* wun = f->derive(f);
	clay::State state = wun->get_state();
	EXPECT_SHAPEQ(builder.shape_, state.shape_);
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

	wire::Identifier* der = f->derive(&arg);
	EXPECT_EQ(&derout, der);

	delete f;
	delete wun;
}


#endif /* DISABLE_OPERATORS_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */
