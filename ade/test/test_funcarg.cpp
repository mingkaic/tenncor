
#ifndef DISABLE_FUNCARG_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "ade/test/common.hpp"

#include "ade/funcarg.hpp"


TEST(FUNCARG, Reduce1d)
{
	size_t rank = 5;
	ade::CoordT fwd_out;
	ade::CoordT icoord = {
		211.6172349153, 3.6941314330, 3.3471187148, 24.3511302088,
		17.8520169468, 99.9911659058, 7.2182000783, 6.4776819746
	};
	ade::Shape shape({223, 35, 7, 25, 19, 214, 72, 7});
	ade::TensptrT tens = std::make_shared<MockTensor>(shape);

	ade::FuncArg redtens = ade::reduce_1d_map(tens, rank);
	ade::Shape rshaped = redtens.shape();

	std::vector<ade::DimT> expect_shape = {223, 35, 7, 25,
		19, 72, 7, 1};
	EXPECT_ARREQ(expect_shape, rshaped);
	EXPECT_TRUE(redtens.map_io());

	auto cmapped = redtens.get_coorder();
	cmapped->forward(fwd_out.begin(), icoord.begin());
	ade::CoordT expect_coord = {
		211.6172349153, 3.6941314330, 3.3471187148, 24.3511302088,
		17.8520169468, 7.2182000783, 6.4776819746, (99.9911659058 / 214.0)
	};
	for (ade::RankT i = 0; i < ade::rank_cap; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_coord[i], fwd_out[i]);
	}
}


#endif // DISABLE_FUNCARG_TEST
