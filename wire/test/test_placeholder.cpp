#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "wire/placeholder.hpp"


#ifndef DISABLE_PLACEHOLDER_TEST


using namespace testutil;


class PLACEHOLDER : public fuzz_test
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


TEST_F(PLACEHOLDER, VecAssign_F000)
{
	std::string label = get_string(16, "label");
	wire::Placeholder place(label);
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	size_t n = get_int(1, "n", {16, 64})[0];
	size_t nbytes = n * clay::type_size(dtype);
	std::string data = get_string(nbytes, "data");

	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
		{
			double* ptr = (double*) data.c_str();
			std::vector<double> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float* ptr = (float*) data.c_str();
			std::vector<float> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t* ptr = (int8_t*) data.c_str();
			std::vector<int8_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t* ptr = (uint8_t*) data.c_str();
			std::vector<uint8_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t* ptr = (int16_t*) data.c_str();
			std::vector<int16_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t* ptr = (uint16_t*) data.c_str();
			std::vector<uint16_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t* ptr = (int32_t*) data.c_str();
			std::vector<int32_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t* ptr = (uint32_t*) data.c_str();
			std::vector<uint32_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t* ptr = (int64_t*) data.c_str();
			std::vector<int64_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t* ptr = (uint64_t*) data.c_str();
			std::vector<uint64_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		default:
			ASSERT_FALSE(true);
	}
	clay::State state = place.get_state();
	EXPECT_EQ(dtype, state.dtype_);
	clay::Shape vshape({n});
	ASSERT_SHAPEQ(vshape, state.shape_);
	std::string gotvec(state.data_.lock().get(), nbytes);
	EXPECT_STREQ(data.c_str(), gotvec.c_str());
}


TEST_F(PLACEHOLDER, ShapedAssign_F000)
{
	std::string label = get_string(16, "label");
	clay::Shape shape = random_def_shape(this);
	wire::Placeholder place(shape, label);
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	size_t n = shape.n_elems();
	size_t nbytes = n * clay::type_size(dtype);
	std::string data = get_string(nbytes, "data");

	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
		{
			double* ptr = (double*) data.c_str();
			std::vector<double> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float* ptr = (float*) data.c_str();
			std::vector<float> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t* ptr = (int8_t*) data.c_str();
			std::vector<int8_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t* ptr = (uint8_t*) data.c_str();
			std::vector<uint8_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t* ptr = (int16_t*) data.c_str();
			std::vector<int16_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t* ptr = (uint16_t*) data.c_str();
			std::vector<uint16_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t* ptr = (int32_t*) data.c_str();
			std::vector<int32_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t* ptr = (uint32_t*) data.c_str();
			std::vector<uint32_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t* ptr = (int64_t*) data.c_str();
			std::vector<int64_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t* ptr = (uint64_t*) data.c_str();
			std::vector<uint64_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		default:
			ASSERT_FALSE(true);
	}
	clay::State state = place.get_state();
	EXPECT_EQ(dtype, state.dtype_);
	ASSERT_SHAPEQ(shape, state.shape_);
	std::string got(state.data_.lock().get(), nbytes);
	EXPECT_STREQ(data.c_str(), got.c_str());
}


TEST_F(PLACEHOLDER, PartAssign_F000)
{
	std::string label = get_string(16, "label");
	std::vector<size_t> clist = random_def_shape(this);
	clay::Shape shape = clist;
	clay::Shape parts = make_partial(this, clist);
	wire::Placeholder place(parts, label);
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	size_t n = shape.n_elems();
	size_t nbytes = n * clay::type_size(dtype);
	std::string data = get_string(nbytes, "data");

	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
		{
			double* ptr = (double*) data.c_str();
			std::vector<double> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float* ptr = (float*) data.c_str();
			std::vector<float> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t* ptr = (int8_t*) data.c_str();
			std::vector<int8_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t* ptr = (uint8_t*) data.c_str();
			std::vector<uint8_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t* ptr = (int16_t*) data.c_str();
			std::vector<int16_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t* ptr = (uint16_t*) data.c_str();
			std::vector<uint16_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t* ptr = (int32_t*) data.c_str();
			std::vector<int32_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t* ptr = (uint32_t*) data.c_str();
			std::vector<uint32_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t* ptr = (int64_t*) data.c_str();
			std::vector<int64_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t* ptr = (uint64_t*) data.c_str();
			std::vector<uint64_t> vec(ptr, ptr + n);
			place = vec;
		}
		break;
		default:
			ASSERT_FALSE(true);
	}
	clay::State state = place.get_state();
	EXPECT_EQ(dtype, state.dtype_);
	ASSERT_EQ(n, state.shape_.n_elems());
	std::string got(state.data_.lock().get(), nbytes);
	EXPECT_STREQ(data.c_str(), got.c_str());
}


TEST_F(PLACEHOLDER, Reassign_F000)
{
	std::string label = get_string(16, "label");
	clay::Shape shape = random_def_shape(this);
	wire::Placeholder place(shape, label);
	bool doub = get_int(1, "doub")[0] % 2;
	size_t n = shape.n_elems();
	size_t bsize;
	if (doub)
	{
		bsize = clay::type_size(clay::DTYPE::DOUBLE);
	}
	else
	{
		bsize = clay::type_size(clay::DTYPE::UINT16);
	}
	size_t nbytes = n * bsize;
	std::string data = get_string(nbytes, "data");

	std::string good = get_string(nbytes, "good");

	size_t less = n - get_int(1, "n-less", {1, n/2})[0];
	size_t lessbytes = less * bsize;
	std::string goodfit = get_string(lessbytes, "goodfit");

	size_t more = n + get_int(1, "n-more", {1, n/2})[0];
	size_t morebytes = more * bsize;
	std::string badfit = get_string(morebytes, "badfit");

	size_t badbsize;
	if (doub)
	{
		badbsize = clay::type_size(clay::DTYPE::UINT16);
	}
	else
	{
		badbsize = clay::type_size(clay::DTYPE::DOUBLE);
	}
	size_t typebytes = n * badbsize;
	std::string badtype = get_string(typebytes, "badtype");

	if (doub)
	{
		double* ptr = (double*) data.c_str();
		std::vector<double> vec(ptr, ptr + n);
		place = vec;

		double* goodptr = (double*) good.c_str();
		std::vector<double> goodvec(goodptr, goodptr + n);
		place = goodvec;
		clay::State state = place.get_state();
		std::string got(state.data_.lock().get(), nbytes);
		EXPECT_STREQ(good.c_str(), got.c_str());

		double* fitptr = (double*) goodfit.c_str();
		std::vector<double> fitvec(fitptr, fitptr + less);
		place = fitvec;
		state = place.get_state();
		std::string gotfit(state.data_.lock().get(), lessbytes);
		EXPECT_STREQ(goodfit.c_str(), gotfit.c_str());

		double* badptr = (double*) badfit.c_str();
		std::vector<double> badvec(badptr, badptr + more);
		EXPECT_THROW(place = badvec, std::exception);

		uint16_t* typeptr = (uint16_t*) badtype.c_str();
		std::vector<uint16_t> typevec(typeptr, typeptr + typebytes);
		EXPECT_THROW(place = typevec, std::exception);
	}
	else
	{
		uint16_t* ptr = (uint16_t*) data.c_str();
		std::vector<uint16_t> vec(ptr, ptr + n);
		place = vec;

		uint16_t* goodptr = (uint16_t*) good.c_str();
		std::vector<uint16_t> goodvec(goodptr, goodptr + n);
		place = goodvec;
		clay::State state = place.get_state();
		std::string got(state.data_.lock().get(), nbytes);
		EXPECT_STREQ(good.c_str(), got.c_str());

		uint16_t* fitptr = (uint16_t*) goodfit.c_str();
		std::vector<uint16_t> fitvec(fitptr, fitptr + less);
		place = fitvec;
		state = place.get_state();
		std::string gotfit(state.data_.lock().get(), lessbytes);
		EXPECT_STREQ(goodfit.c_str(), gotfit.c_str());

		uint16_t* badptr = (uint16_t*) badfit.c_str();
		std::vector<uint16_t> badvec(badptr, badptr + more);
		EXPECT_THROW(place = badvec, std::exception);

		double* typeptr = (double*) badtype.c_str();
		std::vector<double> typevec(typeptr, typeptr + typebytes);
		EXPECT_THROW(place = typevec, std::exception);
	}
}


TEST_F(PLACEHOLDER, Derive_F004)
{
	clay::Shape shape = random_def_shape(this);
	wire::Placeholder var(shape, get_string(16, "plname"));
	wire::Placeholder var2(shape, get_string(16, "plname"));
	std::vector<double> data = get_double(shape.n_elems(), "data");

	EXPECT_THROW(var.derive(&var), std::exception);
	var = data;
	wire::Identifier* wun = var.derive(&var);
	wire::Identifier* zaro = var.derive(&var2);
	clay::Shape scalars(std::vector<size_t>{1});
	clay::State state = wun->get_state();
	clay::State state2 = zaro->get_state();
	EXPECT_SHAPEQ(scalars, state.shape_);
	EXPECT_SHAPEQ(scalars, state2.shape_);

	double scalarw = *((double*) state.data_.lock().get());
	double scalarz = *((double*) state2.data_.lock().get());
	EXPECT_EQ(1, scalarw);
	EXPECT_EQ(0, scalarz);

	delete zaro;
	delete wun;
}


#endif /* DISABLE_PLACEHOLDER_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */
