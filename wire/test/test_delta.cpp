#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "clay/error.hpp"

#include "wire/constant.hpp"
#include "wire/placeholder.hpp"
#include "wire/variable.hpp"
#include "wire/delta.hpp"


#ifndef DISABLE_DELTA_TEST


using namespace testutil;


class DELTA : public fuzz_test
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


TEST_F(DELTA, Constant_H000)
{
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	unsigned short bsize = clay::type_size(dtype);
	std::shared_ptr<char> ptr = clay::make_char(bsize);
	wire::Placeholder c2("");
	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
		{
			double d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2 = std::vector<double>{d};
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2 = std::vector<float>{d};
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2 = std::vector<int8_t>{d};
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2 = std::vector<uint8_t>{d};
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2 = std::vector<int16_t>{d};
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2 = std::vector<uint16_t>{d};
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2 = std::vector<int32_t>{d};
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2 = std::vector<uint32_t>{d};
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2 = std::vector<int64_t>{d};
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2 = std::vector<uint64_t>{d};
		}
		break;
		default:
			ASSERT_TRUE(false) << "generated bad type";
		break;
	}
	wire::Constant c(ptr, clay::Shape({1}), dtype, get_string(16, "cname"));
	EXPECT_THROW(wire::delta(&c, &c), std::logic_error);
	wire::Identifier* zaro = wire::delta(&c, &c2);
	clay::State z = zaro->get_state();
	EXPECT_EQ(dtype, z.dtype_);
	std::vector<size_t> wun{1};
	EXPECT_ARREQ(wun, z.shape_.as_list());
	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
		{
			double gotz = *((double*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float gotz = *((float*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t gotz = *((int8_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t gotz = *((uint8_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t gotz = *((int16_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t gotz = *((uint16_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t gotz = *((int32_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t gotz = *((uint32_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t gotz = *((int64_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t gotz = *((uint64_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		default:
		break;
	}
	delete zaro;
}


TEST_F(DELTA, Variable_H001)
{
	mock_builder builder(this);
	wire::Variable var(builder, get_string(16, "varname"));
	wire::Variable var2(builder, get_string(16, "var2name"));

	EXPECT_THROW(wire::delta(&var, &var), mold::UninitializedError);
	wire::Graph::get_global().initialize_all();
	wire::Identifier* wun = wire::delta(&var, &var);
	wire::Identifier* zaro = wire::delta(&var, &var2);
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


TEST_F(DELTA, Placeholder_H002)
{
	clay::Shape shape = random_def_shape(this);
	wire::Placeholder var(shape, get_string(16, "plname"));
	wire::Placeholder var2(shape, get_string(16, "plname"));
	std::vector<double> data = get_double(shape.n_elems(), "data");

	EXPECT_THROW(wire::delta(&var, &var), mold::UninitializedError);
	var = data;
	wire::Identifier* wun = wire::delta(&var, &var);
	EXPECT_THROW(wire::delta(&var, &var2), mold::UninitializedError);
	var2 = data;
	wire::Identifier* zaro = wire::delta(&var, &var2);
	clay::State state = wun->get_state();
	clay::State state2 = zaro->get_state();
	EXPECT_SHAPEQ(shape, state.shape_);
	EXPECT_SHAPEQ(shape, state2.shape_);

	double scalarw = *((double*) state.data_.lock().get());
	double scalarz = *((double*) state2.data_.lock().get());
	EXPECT_EQ(1, scalarw);
	EXPECT_EQ(0, scalarz);

	delete zaro;
	delete wun;
}


#endif /* DISABLE_DELTA_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */