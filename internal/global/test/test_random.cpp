
#ifndef DISABLE_GLOBAL_RANDOM_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"


TEST(RANDOM, SetGet)
{
	global::CfgMapptrT ctx = std::make_shared<estd::ConfigMap<>>();

	auto& rengine = global::get_randengine(ctx);
	void* origptr = &rengine;
	auto orengine = new global::RandEngineT();
	global::set_randengine(orengine, ctx);
	EXPECT_NE(origptr, &global::get_randengine(ctx));
	EXPECT_EQ(orengine, &global::get_randengine(ctx));

	auto& uengine = global::get_uuidengine(ctx);
	void* origptr2 = &uengine;
	auto ouengine = new global::UuidEngineT();
	global::set_uuidengine(ouengine, ctx);
	EXPECT_NE(origptr2, &global::get_uuidengine(ctx));
	EXPECT_EQ(ouengine, &global::get_uuidengine(ctx));
}


TEST(RANDOM, Randomizer)
{
	global::CfgMapptrT ctx = std::make_shared<estd::ConfigMap<>>();
	global::Randomizer rand(ctx);
	global::seed(0, ctx);

	auto inum = rand.unif<size_t>(2, 7);
	EXPECT_LE(2, inum);
	EXPECT_GE(7, inum);

	auto fnum = rand.unif<float>(2, 7);
	EXPECT_LE(2, fnum);
	EXPECT_GE(7, fnum);

	auto igen = rand.unif_gen<size_t>(2, 7);
	EXPECT_LE(2, igen());
	EXPECT_GE(7, igen());

	auto fgen = rand.unif_gen<float>(2, 7);
	EXPECT_LE(2, fgen());
	EXPECT_GE(7, fgen());
}


#endif // DISABLE_GLOBAL_RANDOM_TEST
