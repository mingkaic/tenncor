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
#include "kiln/const_init.hpp"


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
			c2.initialize(std::vector<double>{d});
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2.initialize(std::vector<float>{d});
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2.initialize(std::vector<int8_t>{d});
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2.initialize(std::vector<uint8_t>{d});
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2.initialize(std::vector<int16_t>{d});
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2.initialize(std::vector<uint16_t>{d});
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2.initialize(std::vector<int32_t>{d});
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2.initialize(std::vector<uint32_t>{d});
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2.initialize(std::vector<int64_t>{d});
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
			c2.initialize(std::vector<uint64_t>{d});
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
	clay::Shape shape = random_def_shape(this);
	clay::BuildTensorT builder = kiln::const_init((double) 0, shape);
	wire::Variable var(builder, get_string(16, "varname"));
	wire::Variable var2(builder, get_string(16, "var2name"));

	EXPECT_THROW(wire::delta(&var, &var), mold::UninitializedError);
	wire::Graph::get_global().initialize_all();
	wire::Identifier* wun = wire::delta(&var, &var);
	wire::Identifier* zaro = wire::delta(&var, &var2);
	clay::State state = wun->get_state();
	clay::State state2 = zaro->get_state();
	EXPECT_SHAPEQ(shape, state.shape_);
	EXPECT_SHAPEQ(shape, state2.shape_);
	EXPECT_EQ(clay::DOUBLE, state.dtype_);
	EXPECT_EQ(clay::DOUBLE, state2.dtype_);
	double scalarw = *((double*) state.data_.lock().get());
	double scalarz = *((double*) state2.data_.lock().get());
	EXPECT_EQ(1, scalarw);
	EXPECT_EQ(0, scalarz);

	delete zaro;
	delete wun;
}


TEST_F(DELTA, Placeholder_H002)
{
	clay::Shape shape = random_def_shape(this);
	wire::Placeholder var(get_string(16, "plname"));
	wire::Placeholder var2(get_string(16, "plname"));
	std::vector<double> data = get_double(shape.n_elems(), "data");

	EXPECT_THROW(wire::delta(&var, &var), mold::UninitializedError);
	var.initialize(data, shape);
	wire::Identifier* wun = wire::delta(&var, &var);
	EXPECT_THROW(wire::delta(&var, &var2), mold::UninitializedError);
	var2.initialize(data, shape);
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