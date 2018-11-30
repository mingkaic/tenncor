
#ifndef DISABLE_CAPI_TEST


#include "gtest/gtest.h"

#include "age/test/grader_dep.hpp"
#include "age/generated/capi.hpp"


TEST(AGE, CApi)
{
	// everything should be exactly the same as Api
	// except inputs and output types are different
	int64_t carrot = goku(16);
	MockTensor* kakarot = dynamic_cast<MockTensor*>(
		static_cast<ade::iTensor*>(get_ptr(carrot)));
	EXPECT_NE(nullptr, kakarot);
	ade::Shape shape = kakarot->shape();
	EXPECT_EQ(16, kakarot->scalar_);
	EXPECT_EQ(16, shape.n_elems());
	EXPECT_EQ(16, shape.at(0));

	int64_t var = malloc_tens(new MockTensor(1, ade::Shape({1, 1, 31})));
	int64_t vegetable = vegeta(var, 2);
	MockTensor* planet = dynamic_cast<MockTensor*>(
		static_cast<ade::iTensor*>(get_ptr(vegetable)));
	EXPECT_NE(nullptr, planet);
	ade::Shape vshape = planet->shape();
	EXPECT_EQ(2, planet->scalar_);
	EXPECT_EQ(31, vshape.n_elems());
	EXPECT_EQ(31, vshape.at(0));
}


#endif // DISABLE_CAPI_TEST
