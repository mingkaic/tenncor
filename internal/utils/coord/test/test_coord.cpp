
#ifndef DISABLE_UTILS_COORD_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/utils/coord/coord.hpp"


using ::testing::_;
using ::testing::Return;
using ::testing::Throw;


TEST(SHAPE, Coordinates)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillRepeatedly(Return(true));

	teq::DimsT slist = {9, 3, 7, 8, 5};
	teq::Shape shape(slist);
	teq::CoordT coord;
	for (teq::NElemT i = 0, n = shape.n_elems(); i < n; ++i)
	{
		coord = teq::coordinate(shape, i);
		for (teq::RankT i = 0; i < teq::rank_cap; ++i)
		{
			EXPECT_GT(shape.at(i), coord[i]);
		}
		teq::NElemT idx = teq::index(shape, coord);
		EXPECT_EQ(i, idx);
	}

	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		coord[i] = shape.at(i);
	}
	std::string shapestr = shape.to_string();
	std::string fatalmsg = fmts::sprintf("cannot get index of bad coordinate "
		"%s for shape %s", shapestr.c_str(), shapestr.c_str());
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(teq::index(shape, coord), fatalmsg.c_str());

	std::string fatalmsg2 = fmts::sprintf("cannot get coordinate of index %d "
		"(>= shape %s)", shape.n_elems(), shapestr.c_str());
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg2, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg2)));
	EXPECT_FATAL(teq::coordinate(shape, shape.n_elems()), fatalmsg2.c_str());

	global::set_logger(new exam::NoSupportLogger());
}


#endif // DISABLE_UTILS_COORD_TEST
