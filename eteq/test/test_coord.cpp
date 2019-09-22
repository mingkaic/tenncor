
#ifndef DISABLE_COORD_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "eteq/eteq.hpp"


TEST(COORD, Connect)
{
	eteq::CoordMap cmap({1, 2, 3, 4, 5, 6, 7, 8}, false);
	eteq::CoordMap other({9, 8, 7, 6, 5, 4, 3, 2}, true);
	EXPECT_EQ(nullptr, cmap.connect(other));
}


TEST(COORD, Forward)
{
	teq::CoordT expect_a = {1, 2, 3, 4, 5, 6, 7, 8};
	teq::CoordT expect_b = {9, 8, 7, 6, 5, 4, 3, 2};

	eteq::CoordMap a(expect_a, false);
	eteq::CoordMap b(expect_b, true);

	teq::CoordT out;
	a.forward(out.begin(), out.begin());
	EXPECT_ARREQ(expect_a, out);

	b.forward(out.begin(), out.begin());
	EXPECT_ARREQ(expect_b, out);
}


TEST(COORD, Reverse)
{
	eteq::CoordMap cmap({1, 2, 3, 4, 5, 6, 7, 8}, false);
	EXPECT_EQ(nullptr, cmap.reverse());
}


#endif // DISABLE_COORD_TEST
