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


#endif /* DISABLE_VARIABLE_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */
