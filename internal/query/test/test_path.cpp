
#ifndef DISABLE_PATH_TEST


#include "gtest/gtest.h"

#include "internal/teq/mock/leaf.hpp"
#include "internal/teq/mock/functor.hpp"

#include "internal/query/path.hpp"


TEST(PATH, GetArgs)
{
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	query::Path path(x.get());
	auto args = path.get_args();
	EXPECT_EQ(0, args.size());

	auto arg1 = std::make_shared<MockFunctor>(teq::TensptrsT{x}, teq::Opcode{"SIN", 0});
	auto arg2 = std::make_shared<MockFunctor>(teq::TensptrsT{x,x}, teq::Opcode{"SUB", 0});
	MockFunctor f(teq::TensptrsT{arg1, arg2});
	query::Path path2(&f);
	auto args2 = path2.get_args();
	ASSERT_EQ(2, args2.size());
	EXPECT_EQ(0, args2[0].first);
	EXPECT_EQ(1, args2[1].first);
	EXPECT_EQ(arg1.get(), args2[0].second);
	EXPECT_EQ(arg2.get(), args2[1].second);
}


TEST(PATH, Recall)
{
	MockLeaf x(teq::Shape(), "X");
	auto noprev = std::make_shared<query::Path>(&x);
	EXPECT_EQ(nullptr, noprev->recall());

	query::Path hasprev(&x, {3, noprev});
	EXPECT_NE(noprev, hasprev.recall());
	EXPECT_EQ(&x, hasprev.recall()->tens_);
}


#endif // DISABLE_PATH_TEST
