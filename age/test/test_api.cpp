
#ifndef DISABLE_API_TEST


#include "gtest/gtest.h"

#include "age/test/grader_dep.hpp"
#include "age/generated/api.hpp"


TEST(AGE, Api)
{
    ade::Tensorptr carrot = age::goku(16);
    MockTensor* kakarot = dynamic_cast<MockTensor*>(carrot.get());
    EXPECT_NE(nullptr, kakarot);
    ade::Shape shape = kakarot->shape();
    EXPECT_EQ(16, kakarot->scalar_);
    EXPECT_EQ(16, shape.n_elems());
    EXPECT_EQ(16, shape.at(0));
}


#endif // DISABLE_API_TEST
