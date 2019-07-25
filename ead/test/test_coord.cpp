
#ifndef DISABLE_COORD_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "ead/ead.hpp"


TEST(COORD, Connect)
{
	ead::CoordMap cmap({1, 2, 3, 4, 5, 6, 7, 8}, false);
	ead::CoordMap other({9, 8, 7, 6, 5, 4, 3, 2}, true);
	EXPECT_EQ(nullptr, cmap.connect(other));
}


TEST(COORD, Forward)
{
	ade::CoordT expect_a = {1, 2, 3, 4, 5, 6, 7, 8};
	ade::CoordT expect_b = {9, 8, 7, 6, 5, 4, 3, 2};

	ead::CoordMap a(expect_a, false);
	ead::CoordMap b(expect_b, true);

	ade::CoordT out;
	a.forward(out.begin(), out.begin());
	EXPECT_ARREQ(expect_a, out);

	b.forward(out.begin(), out.begin());
	EXPECT_ARREQ(expect_b, out);
}


TEST(COORD, Reverse)
{
	ead::CoordMap cmap({1, 2, 3, 4, 5, 6, 7, 8}, false);
	EXPECT_EQ(nullptr, cmap.reverse());
}


#endif // DISABLE_COORD_TEST
