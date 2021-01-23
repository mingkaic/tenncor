
#ifndef DISABLE_QUERY_PATH_TEST


#include "gtest/gtest.h"

#include "internal/teq/mock/mock.hpp"

#include "internal/query/path.hpp"


using ::testing::_;
using ::testing::Return;
using ::testing::Const;


TEST(PATH, GetArgs)
{
	auto x = make_var(teq::Shape(), "X");
	query::Path path(x.get());
	auto args = path.get_args();
	EXPECT_EQ(0, args.size());

	auto arg1 = make_fnc("SIN", 0, teq::TensptrsT{x});
	auto arg2 = make_fnc("SUB", 0, teq::TensptrsT{x,x});
	auto f = make_fnc("", 0, teq::TensptrsT{arg1, arg2});
	query::Path path2(f.get());
	auto args2 = path2.get_args();
	ASSERT_EQ(2, args2.size());
	EXPECT_EQ(0, args2[0].first);
	EXPECT_EQ(1, args2[1].first);
	EXPECT_EQ(arg1.get(), args2[0].second);
	EXPECT_EQ(arg2.get(), args2[1].second);
}


TEST(PATH, Recall)
{
	MockLeaf x;
	make_var(x, teq::Shape(), "X");
	EXPECT_CALL(x, get_usage()).WillRepeatedly(Return(teq::IMMUTABLE));

	auto noprev = std::make_shared<query::Path>(&x);
	EXPECT_EQ(nullptr, noprev->recall());

	query::Path hasprev(&x, {3, noprev});
	EXPECT_NE(noprev, hasprev.recall());
	EXPECT_EQ(&x, hasprev.recall()->tens_);
}


#endif // DISABLE_QUERY_PATH_TEST
