
#ifndef DISABLE_COORD_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "eteq/eteq.hpp"


TEST(COORD, Connect)
{
	eigen::CoordMap cmap({1, 2, 3, 4, 5, 6, 7, 8}, false);
	eigen::CoordMap other({9, 8, 7, 6, 5, 4, 3, 2}, true);
	EXPECT_EQ(nullptr, cmap.connect(other));
}


TEST(COORD, Forward)
{
	teq::CoordT expect_a = {1, 2, 3, 4, 5, 6, 7, 8};
	teq::CoordT expect_b = {9, 8, 7, 6, 5, 4, 3, 2};

	eigen::CoordMap a(expect_a);
	eigen::CoordMap b(expect_b, true);

	teq::CoordT out;
	a.access(
		[&](const teq::MatrixT& args)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				out[i] = args[0][i];
			}
		});
	EXPECT_ARREQ(expect_a, out);

	b.access(
		[&](const teq::MatrixT& args)
		{
			for (teq::RankT i = 0; i < teq::rank_cap; ++i)
			{
				out[i] = args[0][i];
			}
		});
	EXPECT_ARREQ(expect_b, out);
}


TEST(COORD, Reverse)
{
	eigen::CoordMap cmap({1, 2, 3, 4, 5, 6, 7, 8}, false);
	EXPECT_EQ(nullptr, cmap.reverse());
}


#endif // DISABLE_COORD_TEST
