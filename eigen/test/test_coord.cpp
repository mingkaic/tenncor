
#ifndef DISABLE_COORD_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "eigen/coord.hpp"


TEST(COORD, Forward)
{
	teq::CoordT expect_a = {1, 2, 3, 4, 5, 6, 7, 8};
	teq::CoordT expect_b = {9, 8, 7, 6, 5, 4, 3, 2};

	eigen::CoordMap a(expect_a);
	eigen::CoordMap b(expect_b);

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


#endif // DISABLE_COORD_TEST
