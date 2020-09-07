
#ifndef DISABLE_GLOBAL_RANDOM_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"


TEST(RANDOM, SetGet)
{
	global::CfgMapptrT ctx = std::make_shared<estd::ConfigMap<>>();

	auto generator = global::get_generator(ctx);
	void* origptr = generator.get();
	auto ogenerator = std::make_shared<global::Randomizer>();
	global::set_generator(ogenerator, ctx);
	EXPECT_NE(origptr, global::get_generator(ctx).get());
	EXPECT_EQ(ogenerator.get(), global::get_generator(ctx).get());
}


TEST(RANDOM, Randomizer)
{
	global::Randomizer rand;
	rand.seed(0);

	auto inum = rand.unif_int(2, 7);
	EXPECT_LE(2, inum);
	EXPECT_GE(7, inum);

	auto fnum = rand.unif_dec(2, 7);
	EXPECT_LE(2, fnum);
	EXPECT_GE(7, fnum);

	auto igen = rand.unif_intgen(2, 7);
	EXPECT_LE(2, igen());
	EXPECT_GE(7, igen());

	auto fgen = rand.unif_decgen(2, 7);
	EXPECT_LE(2, fgen());
	EXPECT_GE(7, fgen());
}


#endif // DISABLE_GLOBAL_RANDOM_TEST
