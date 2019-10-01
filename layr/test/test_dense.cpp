
#ifndef DISABLE_DENSE_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "layr/dense.hpp"


TEST(DENSE, Copy)
{
	std::string label = "especially_dense";
	std::string rlabel = "kinda_dense";
	std::string nb_label = "fake_news";
	layr::Dense dense(4, 5,
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		label);
	layr::Dense rdense(5, 6,
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		rlabel);
	layr::Dense nobias(6, 7,
		layr::unif_xavier_init<PybindT>(3),
		layr::InitF<PybindT>(),
		nb_label);

	auto exdense = dense.get_contents();
	auto exrdense = rdense.get_contents();
	auto exnobias = nobias.get_contents();

	layr::Dense cpyd(dense);
	layr::Dense cpyr(rdense);
	layr::Dense cpyn(nobias);

	EXPECT_STREQ(label.c_str(), cpyd.get_label().c_str());
	EXPECT_STREQ(rlabel.c_str(), cpyr.get_label().c_str());
	EXPECT_STREQ(nb_label.c_str(), cpyn.get_label().c_str());

	EXPECT_EQ(5, cpyd.get_ninput());
	EXPECT_EQ(6, cpyr.get_ninput());
	EXPECT_EQ(7, cpyn.get_ninput());

	EXPECT_EQ(4, cpyd.get_noutput());
	EXPECT_EQ(5, cpyr.get_noutput());
	EXPECT_EQ(6, cpyn.get_noutput());

	auto gotdense = cpyd.get_contents();
	auto gotrdense = cpyr.get_contents();
	auto gonobias = cpyn.get_contents();

	ASSERT_EQ(exdense.size(), gotdense.size());
	ASSERT_EQ(2, gotdense.size());
	ASSERT_EQ(exrdense.size(), gotrdense.size());
	ASSERT_EQ(2, gotrdense.size());
	ASSERT_EQ(exnobias.size(), gonobias.size());
	ASSERT_EQ(2, gonobias.size());

	ASSERT_NE(exdense[0], gotdense[0]);
	ASSERT_NE(exdense[1], gotdense[1]);
	EXPECT_STREQ("weight", gotdense[0]->to_string().c_str());
	EXPECT_STREQ("bias", gotdense[1]->to_string().c_str());
	EXPECT_TENSDATA(exdense[0].get(), gotdense[0].get(), PybindT);
	EXPECT_TENSDATA(exdense[1].get(), gotdense[1].get(), PybindT);

	ASSERT_NE(exrdense[0], gotrdense[0]);
	ASSERT_NE(exrdense[1], gotrdense[1]);
	EXPECT_STREQ("weight", gotrdense[0]->to_string().c_str());
	EXPECT_STREQ("bias", gotrdense[1]->to_string().c_str());
	EXPECT_TENSDATA(exrdense[0].get(), gotrdense[0].get(), PybindT);
	EXPECT_TENSDATA(exrdense[1].get(), gotrdense[1].get(), PybindT);

	ASSERT_NE(exnobias[0], gonobias[0]);
	ASSERT_EQ(exnobias[1], gonobias[1]);
	ASSERT_EQ(nullptr, gonobias[1]);
	EXPECT_STREQ("weight", gonobias[0]->to_string().c_str());
	EXPECT_TENSDATA(exnobias[0].get(), gonobias[0].get(), PybindT);
}


TEST(DENSE, Clone)
{
	std::string label = "especially_dense";
	std::string rlabel = "kinda_dense";
	std::string nb_label = "fake_news";
	layr::Dense dense(4, 5,
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		label);
	layr::Dense rdense(5, 6,
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		rlabel);
	layr::Dense nobias(6, 7,
		layr::unif_xavier_init<PybindT>(3),
		layr::InitF<PybindT>(),
		nb_label);

	layr::Dense* cpyd = dense.clone("ghi");
	layr::Dense* cpyr = rdense.clone("def");
	layr::Dense* cpyn = nobias.clone("abc");

	EXPECT_STREQ(("ghi" + label).c_str(), cpyd->get_label().c_str());
	EXPECT_STREQ(("def" + rlabel).c_str(), cpyr->get_label().c_str());
	EXPECT_STREQ(("abc" + nb_label).c_str(), cpyn->get_label().c_str());

	delete cpyd;
	delete cpyr;
	delete cpyn;
}


TEST(DENSE, Move)
{
	std::string label = "especially_dense";
	std::string rlabel = "kinda_dense";
	std::string nb_label = "fake_news";
	layr::Dense dense(4, 5,
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		label);
	layr::Dense rdense(5, 6,
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		rlabel);
	layr::Dense nobias(6, 7,
		layr::unif_xavier_init<PybindT>(3),
		layr::InitF<PybindT>(),
		nb_label);

	auto exdense = dense.get_contents();
	auto exrdense = rdense.get_contents();
	auto exnobias = nobias.get_contents();

	layr::Dense mvd(std::move(dense));
	layr::Dense mvr(std::move(rdense));
	layr::Dense mvn(std::move(nobias));

	EXPECT_STREQ("", dense.get_label().c_str());
	EXPECT_STREQ("", rdense.get_label().c_str());
	EXPECT_STREQ("", nobias.get_label().c_str());
	EXPECT_STREQ(label.c_str(), mvd.get_label().c_str());
	EXPECT_STREQ(rlabel.c_str(), mvr.get_label().c_str());
	EXPECT_STREQ(nb_label.c_str(), mvn.get_label().c_str());

	EXPECT_EQ(5, mvd.get_ninput());
	EXPECT_EQ(6, mvr.get_ninput());
	EXPECT_EQ(7, mvn.get_ninput());

	EXPECT_EQ(4, mvd.get_noutput());
	EXPECT_EQ(5, mvr.get_noutput());
	EXPECT_EQ(6, mvn.get_noutput());

	auto gotdense = mvd.get_contents();
	auto gotrdense = mvr.get_contents();
	auto gonobias = mvn.get_contents();

	ASSERT_EQ(exdense.size(), gotdense.size());
	ASSERT_EQ(2, gotdense.size());
	ASSERT_EQ(exrdense.size(), gotrdense.size());
	ASSERT_EQ(2, gotrdense.size());
	ASSERT_EQ(exnobias.size(), gonobias.size());
	ASSERT_EQ(2, gonobias.size());

	ASSERT_EQ(exdense[0], gotdense[0]);
	ASSERT_EQ(exdense[1], gotdense[1]);

	ASSERT_EQ(exrdense[0], gotrdense[0]);
	ASSERT_EQ(exrdense[1], gotrdense[1]);

	ASSERT_EQ(exnobias[0], gonobias[0]);
	ASSERT_EQ(nullptr, exnobias[1]);
	ASSERT_EQ(nullptr, gonobias[1]);
}


TEST(DENSE, Connection)
{
	//
}


TEST(DENSE, Tagging)
{
	//
}


TEST(DENSE, Building)
{
	//
}


TEST(DENSE, ConnectionTagging)
{
	//
}


#endif // DISABLE_DENSE_TEST
