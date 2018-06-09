#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "mold/iobserver.hpp"
#include "mold/error.hpp"

#include "wire/variable.hpp"
#include "kiln/const_init.hpp"


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


TEST_F(VARIABLE, Init_E000)
{
	wire::Graph& graph = wire::Graph::get_global();
	EXPECT_EQ(0, graph.n_uninit());

	double scalar = get_double(1, "scalar")[0];
	clay::Shape shape = random_def_shape(this);
	clay::BuildTensorT builder = kiln::const_init(scalar, shape);
	std::string label = get_string(16, "label");
	wire::Variable var(builder, label);

	EXPECT_EQ(1, graph.n_uninit());
	graph.initialize(var.get_uid());
	ASSERT_TRUE(var.has_data());
	clay::State state = var.get_state();
	EXPECT_SHAPEQ(shape, state.shape_);

	EXPECT_EQ(0, graph.n_uninit());
}


#endif /* DISABLE_VARIABLE_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */
