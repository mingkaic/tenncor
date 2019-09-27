
#ifndef DISABLE_FUNCARG_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/test/common.hpp"

#include "teq/funcarg.hpp"


TEST(FUNCARG, Reduce1d)
{
	size_t rank = 5;
	teq::CoordT fwd_out;
	teq::CoordT icoord = {
		211.6172349153, 3.6941314330, 3.3471187148, 24.3511302088,
		17.8520169468, 99.9911659058, 7.2182000783, 6.4776819746
	};
	teq::Shape shape({223, 35, 7, 25, 19, 214, 72, 7});
	teq::TensptrT tens = std::make_shared<MockTensor>(shape);

	teq::FuncArg redtens = teq::reduce_1d_map(tens, rank);
	teq::Shape rshaped = redtens.shape();

	std::vector<teq::DimT> expect_shape = {223, 35, 7, 25,
		19, 72, 7, 1};
	EXPECT_ARREQ(expect_shape, rshaped);
	EXPECT_TRUE(redtens.map_io());

	auto cmapped = redtens.get_coorder();
	cmapped->forward(fwd_out.begin(), icoord.begin());
	teq::CoordT expect_coord = {
		211.6172349153, 3.6941314330, 3.3471187148, 24.3511302088,
		17.8520169468, 7.2182000783, 6.4776819746, (99.9911659058 / 214.0)
	};
	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_coord[i], fwd_out[i]);
	}
}


TEST(FUNCARG, Reduce)
{
	size_t rank = 5;
	teq::CoordT fwd_out;
	teq::CoordT icoord = {
		211.6172349153, 3.6941314330, 3.3471187148, 24.3511302088,
		17.8520169468, 99.9911659058, 7.2182000783, 6.4776819746
	};
	teq::Shape shape({223, 35, 7, 25, 19, 214, 72, 7});
	teq::TensptrT tens = std::make_shared<MockTensor>(shape);

	teq::FuncArg redtens = teq::reduce_map(tens, rank, {2});
	teq::Shape rshaped = redtens.shape();

	std::vector<teq::DimT> expect_shape = {223, 35, 7, 25,
		19, 214 / 2, 72, 7};
	EXPECT_ARREQ(expect_shape, rshaped);
	EXPECT_TRUE(redtens.map_io());

	auto cmapped = redtens.get_coorder();
	cmapped->forward(fwd_out.begin(), icoord.begin());
	teq::CoordT expect_coord = {
		211.6172349153, 3.6941314330, 3.3471187148, 24.3511302088,
		17.8520169468, (99.9911659058 / 2), 7.2182000783, 6.4776819746
	};
	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_coord[i], fwd_out[i]);
	}
}


TEST(FUNCARG, Extend)
{
	size_t rank = 5;
	teq::CoordT fwd_out;
	teq::CoordT icoord = {
		211.6172349153, 3.6941314330, 3.3471187148, 24.3511302088,
		17.8520169468, 99.9911659058, 7.2182000783, 6.4776819746
	};
	teq::Shape shape({223, 35, 7, 25, 19, 214, 72, 7});
	teq::TensptrT tens = std::make_shared<MockTensor>(shape);

	teq::FuncArg extens = teq::extend_map(tens, rank, {2, 3});
	teq::Shape eshaped = extens.shape();

	std::vector<teq::DimT> expect_shape = {223, 35, 7, 25,
		19, 214 * 2, 72 * 3, 7};
	EXPECT_ARREQ(expect_shape, eshaped);
	EXPECT_FALSE(extens.map_io());

	auto cmapped = extens.get_coorder();
	cmapped->forward(fwd_out.begin(), icoord.begin());
	teq::CoordT expect_coord = {
		211.6172349153, 3.6941314330, 3.3471187148, 24.3511302088,
		17.8520169468, 99.9911659058 / 2, 7.2182000783 / 3, 6.4776819746
	};
	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_coord[i], fwd_out[i]);
	}
}


TEST(FUNCARG, Permute)
{
	teq::CoordT fwd_out;
	teq::CoordT icoord = {
		211.6172349153, 3.6941314330, 3.3471187148, 24.3511302088,
		17.8520169468, 99.9911659058, 7.2182000783, 6.4776819746
	};
	teq::Shape shape({223, 35, 7, 25, 19, 214, 72, 7});
	teq::TensptrT tens = std::make_shared<MockTensor>(shape);

	teq::FuncArg ptens = teq::permute_map(tens, {3, 5, 2});
	teq::Shape pshaped = ptens.shape();

	std::vector<teq::DimT> expect_shape = {25, 214, 7, 223,
		35, 19, 72, 7};
	EXPECT_ARREQ(expect_shape, pshaped);
	EXPECT_FALSE(ptens.map_io());

	auto cmapped = ptens.get_coorder();
	cmapped->forward(fwd_out.begin(), icoord.begin());
	teq::CoordT expect_coord = {
		24.3511302088, 17.8520169468, 3.3471187148, 211.6172349153,
		99.9911659058, 3.6941314330, 7.2182000783, 6.4776819746
	};
	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_coord[i], fwd_out[i]);
	}
}


TEST(FUNCARG, ShapeCoordDiff)
{
	teq::CoordptrT shaper = teq::identity;
	teq::CoordptrT coorder = teq::flip(1);
	teq::Shape shape({3, 2});
	teq::TensptrT tens = std::make_shared<MockTensor>(shape);

	teq::FuncArg farg(tens, shaper, true, coorder);
	teq::CoordT fwd_out;
	teq::CoordT icoord = {
		1, 1, 0, 0, 0, 0, 0, 0,
	};
	teq::Shape fshaped = farg.shape();

	std::vector<teq::DimT> expect_shape = {3, 2, 1, 1, 1, 1, 1, 1};
	EXPECT_ARREQ(expect_shape, fshaped);
	EXPECT_TRUE(farg.map_io());

	auto cmapped = farg.get_coorder();
	cmapped->forward(fwd_out.begin(), icoord.begin());
	teq::CoordT expect_coord = {
		1, -2, 0, 0, 0, 0, 0, 0,
	};
	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		EXPECT_DOUBLE_EQ(expect_coord[i], fwd_out[i]);
	}
}


TEST(FUNCARG, ToArgs)
{
	teq::Shape shape({3, 2});
	teq::TensptrT tens = std::make_shared<MockTensor>(shape);
	teq::TensptrT tens2 = std::make_shared<MockTensor>(shape);
	auto args = teq::to_args({tens, tens2});

	ASSERT_EQ(2, args.size());

	auto arg = args[0];
	EXPECT_EQ(tens, arg.get_tensor());
	EXPECT_EQ(teq::identity, arg.get_shaper());
	EXPECT_EQ(teq::identity, arg.get_coorder());

	auto arg2 = args[1];
	EXPECT_EQ(tens2, arg2.get_tensor());
	EXPECT_EQ(teq::identity, arg2.get_shaper());
	EXPECT_EQ(teq::identity, arg2.get_coorder());
}


#endif // DISABLE_FUNCARG_TEST
