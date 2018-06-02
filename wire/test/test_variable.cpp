#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "clay/memory.hpp"

#include "mold/iobserver.hpp"
#include "mold/error.hpp"

#include "wire/variable.hpp"


#ifndef DISABLE_VARIABLE_TEST


using namespace testutil;


class VARIABLE : public fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		fuzz_test::TearDown();
		wire::Graph& g = wire::Graph::get_global();
		assert(0 == g.size());
	}
};


struct mock_builder final : public clay::iBuilder
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

	mock_builder (const mock_builder& other) :
		shape_(other.shape_), dtype_(other.dtype_),
		uuid_(other.uuid_), ptr_(other.ptr_) {}

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


TEST_F(VARIABLE, Init_E000)
{
	wire::Graph& graph = wire::Graph::get_global();
	EXPECT_EQ(0, graph.n_uninit());

	mock_builder builder(this);
	clay::Shape shape = random_def_shape(this);
	std::string label = get_string(16, "label");
	std::string label2 = get_string(16, "label2");
	wire::Variable var(builder, label);
	wire::Variable var2(builder, shape, label2);

	EXPECT_EQ(2, graph.n_uninit());
	graph.initialize(var.get_uid());
	ASSERT_TRUE(var.has_data());
	clay::State state = var.get_state();
	EXPECT_SHAPEQ(builder.shape_, state.shape_);

	EXPECT_EQ(1, graph.n_uninit());
	graph.initialize(var2.get_uid());
	ASSERT_TRUE(var2.has_data());
	clay::State state2 = var2.get_state();
	EXPECT_SHAPEQ(shape, state2.shape_);

	EXPECT_EQ(0, graph.n_uninit());
}


TEST_F(VARIABLE, Derive_C005)
{
	mock_builder builder(this);
	wire::Variable var(builder, get_string(16, "varname"));
	wire::Variable var2(builder, get_string(16, "var2name"));

	EXPECT_THROW(var.derive(&var), mold::UninitializedError);
	wire::Graph::get_global().initialize_all();
	wire::Identifier* wun = var.derive(&var);
	wire::Identifier* zaro = var.derive(&var2);
	clay::State state = wun->get_state();
	clay::State state2 = zaro->get_state();
	EXPECT_SHAPEQ(builder.shape_, state.shape_);
	EXPECT_SHAPEQ(builder.shape_, state2.shape_);
	EXPECT_EQ(builder.dtype_, state.dtype_);
	EXPECT_EQ(builder.dtype_, state2.dtype_);
	switch (builder.dtype_)
	{
		case clay::DTYPE::DOUBLE:
		{
			double scalarw = *((double*) state.data_.lock().get());
			double scalarz = *((double*) state2.data_.lock().get());
			EXPECT_EQ(1, scalarw);
			EXPECT_EQ(0, scalarz);
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float scalarw = *((float*) state.data_.lock().get());
			float scalarz = *((float*) state2.data_.lock().get());
			EXPECT_EQ(1, scalarw);
			EXPECT_EQ(0, scalarz);
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t scalarw = *((int8_t*) state.data_.lock().get());
			int8_t scalarz = *((int8_t*) state2.data_.lock().get());
			EXPECT_EQ(1, scalarw);
			EXPECT_EQ(0, scalarz);
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t scalarw = *((uint8_t*) state.data_.lock().get());
			uint8_t scalarz = *((uint8_t*) state2.data_.lock().get());
			EXPECT_EQ(1, scalarw);
			EXPECT_EQ(0, scalarz);
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t scalarw = *((int16_t*) state.data_.lock().get());
			int16_t scalarz = *((int16_t*) state2.data_.lock().get());
			EXPECT_EQ(1, scalarw);
			EXPECT_EQ(0, scalarz);
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t scalarw = *((uint16_t*) state.data_.lock().get());
			uint16_t scalarz = *((uint16_t*) state2.data_.lock().get());
			EXPECT_EQ(1, scalarw);
			EXPECT_EQ(0, scalarz);
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t scalarw = *((int32_t*) state.data_.lock().get());
			int32_t scalarz = *((int32_t*) state2.data_.lock().get());
			EXPECT_EQ(1, scalarw);
			EXPECT_EQ(0, scalarz);
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t scalarw = *((uint32_t*) state.data_.lock().get());
			uint32_t scalarz = *((uint32_t*) state2.data_.lock().get());
			EXPECT_EQ(1, scalarw);
			EXPECT_EQ(0, scalarz);
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t scalarw = *((int64_t*) state.data_.lock().get());
			int64_t scalarz = *((int64_t*) state2.data_.lock().get());
			EXPECT_EQ(1, scalarw);
			EXPECT_EQ(0, scalarz);
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t scalarw = *((uint64_t*) state.data_.lock().get());
			uint64_t scalarz = *((uint64_t*) state2.data_.lock().get());
			EXPECT_EQ(1, scalarw);
			EXPECT_EQ(0, scalarz);
		}
		break;
		default:
		break;
	}

	delete zaro;
	delete wun;
}


#endif /* DISABLE_VARIABLE_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */
