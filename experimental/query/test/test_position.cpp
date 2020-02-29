
#ifndef DISABLE_POSITION_TEST


#include "gtest/gtest.h"

#include "teq/mock/leaf.hpp"

#include "query/position.hpp"


TEST(POSITION, Compare)
{
	MockLeaf smol(teq::Shape({5, 1}), "1");
	MockLeaf med(teq::Shape({5, 1}), "2");
	MockLeaf big(teq::Shape({5, 2}), "2");

	query::TensPosition a(&smol, 1);
	query::TensPosition b(&med, 1);
	query::TensPosition c(&big, 1);
	EXPECT_LT(a, b);
	EXPECT_LT(b, c);
	EXPECT_LT(a, c);

	EXPECT_GT(b, a);
	EXPECT_GT(c, b);
	EXPECT_GT(c, a);
}


#endif // DISABLE_POSITION_TEST
