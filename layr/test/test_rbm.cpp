
#ifndef DISABLE_RBM_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "layr/rbm.hpp"


TEST(RBM, Copy)
{
	std::string label = "especially_restrictive";
	std::string rlabel = "kinda_restrictive";
	std::string nb_label = "fake_news";
	layr::RBM rbm(4, 5,
        layr::sigmoid(),
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		label);
	layr::RBM rrbm(5, 6,
        layr::sigmoid(),
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		rlabel);
	layr::RBM nobias(6, 7,
        layr::sigmoid(),
		layr::unif_xavier_init<PybindT>(3),
		layr::InitF<PybindT>(),
		nb_label);

	auto exrbm = rbm.get_contents();
	auto exrrbm = rrbm.get_contents();
	auto exnobias = nobias.get_contents();

	layr::RBM cpyd(rbm);
	layr::RBM cpyr(rrbm);
	layr::RBM cpyn(nobias);

	EXPECT_STREQ(label.c_str(), cpyd.get_label().c_str());
	EXPECT_STREQ(rlabel.c_str(), cpyr.get_label().c_str());
	EXPECT_STREQ(nb_label.c_str(), cpyn.get_label().c_str());

	EXPECT_EQ(5, cpyd.get_ninput());
	EXPECT_EQ(6, cpyr.get_ninput());
	EXPECT_EQ(7, cpyn.get_ninput());

	EXPECT_EQ(4, cpyd.get_noutput());
	EXPECT_EQ(5, cpyr.get_noutput());
	EXPECT_EQ(6, cpyn.get_noutput());

	auto gotrbm = cpyd.get_contents();
	auto gotrrbm = cpyr.get_contents();
	auto gonobias = cpyn.get_contents();

	ASSERT_EQ(exrbm.size(), gotrbm.size());
	ASSERT_EQ(5, gotrbm.size());
	ASSERT_EQ(exrrbm.size(), gotrrbm.size());
	ASSERT_EQ(5, gotrrbm.size());
	ASSERT_EQ(exnobias.size(), gonobias.size());
	ASSERT_EQ(5, gonobias.size());

	ASSERT_NE(exrbm[0], gotrbm[0]);
	ASSERT_NE(exrbm[1], gotrbm[1]);
	ASSERT_NE(exrbm[3], gotrbm[3]);
	EXPECT_STREQ("weight", gotrbm[0]->to_string().c_str());
	EXPECT_STREQ("bias", gotrbm[1]->to_string().c_str());
	EXPECT_STREQ("bias", gotrbm[3]->to_string().c_str());
	EXPECT_TENSDATA(exrbm[0].get(), gotrbm[0].get(), PybindT);
	EXPECT_TENSDATA(exrbm[1].get(), gotrbm[1].get(), PybindT);
	EXPECT_TENSDATA(exrbm[3].get(), gotrbm[3].get(), PybindT);

	ASSERT_NE(exrrbm[0], gotrrbm[0]);
	ASSERT_NE(exrrbm[1], gotrrbm[1]);
	ASSERT_NE(exrrbm[3], gotrrbm[3]);
	EXPECT_STREQ("weight", gotrrbm[0]->to_string().c_str());
	EXPECT_STREQ("bias", gotrrbm[1]->to_string().c_str());
	EXPECT_STREQ("bias", gotrrbm[3]->to_string().c_str());
	EXPECT_TENSDATA(exrrbm[0].get(), gotrrbm[0].get(), PybindT);
	EXPECT_TENSDATA(exrrbm[1].get(), gotrrbm[1].get(), PybindT);
	EXPECT_TENSDATA(exrrbm[3].get(), gotrrbm[3].get(), PybindT);

	ASSERT_NE(exnobias[0], gonobias[0]);
	ASSERT_EQ(exnobias[1], gonobias[1]);
	ASSERT_EQ(exnobias[3], gonobias[3]);
	ASSERT_EQ(nullptr, gonobias[1]);
	ASSERT_EQ(nullptr, gonobias[3]);
	EXPECT_STREQ("weight", gonobias[0]->to_string().c_str());
	EXPECT_TENSDATA(exnobias[0].get(), gonobias[0].get(), PybindT);
}


TEST(RBM, Clone)
{
	std::string label = "especially_restrictive";
	std::string rlabel = "kinda_restrictive";
	std::string nb_label = "fake_news";
	layr::RBM rbm(4, 5,
        layr::sigmoid(),
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		label);
	layr::RBM rrbm(5, 6,
        layr::sigmoid(),
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		rlabel);
	layr::RBM nobias(6, 7,
        layr::sigmoid(),
		layr::unif_xavier_init<PybindT>(3),
		layr::InitF<PybindT>(),
		nb_label);

	layr::RBM* cpyd = rbm.clone("ghi");
	layr::RBM* cpyr = rrbm.clone("def");
	layr::RBM* cpyn = nobias.clone("abc");

	EXPECT_STREQ(("ghi" + label).c_str(), cpyd->get_label().c_str());
	EXPECT_STREQ(("def" + rlabel).c_str(), cpyr->get_label().c_str());
	EXPECT_STREQ(("abc" + nb_label).c_str(), cpyn->get_label().c_str());

	delete cpyd;
	delete cpyr;
	delete cpyn;
}


TEST(RBM, Move)
{
	std::string label = "especially_restrictive";
	std::string rlabel = "kinda_restrictive";
	std::string nb_label = "fake_news";
	layr::RBM rbm(4, 5,
        layr::sigmoid(),
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		label);
	layr::RBM rrbm(5, 6,
        layr::sigmoid(),
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		rlabel);
	layr::RBM nobias(6, 7,
        layr::sigmoid(),
		layr::unif_xavier_init<PybindT>(3),
		layr::InitF<PybindT>(),
		nb_label);

	auto exrbm = rbm.get_contents();
	auto exrrbm = rrbm.get_contents();
	auto exnobias = nobias.get_contents();

	layr::RBM mvd(std::move(rbm));
	layr::RBM mvr(std::move(rrbm));
	layr::RBM mvn(std::move(nobias));

	EXPECT_STREQ("", rbm.get_label().c_str());
	EXPECT_STREQ("", rrbm.get_label().c_str());
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

	auto gotrbm = mvd.get_contents();
	auto gotrrbm = mvr.get_contents();
	auto gonobias = mvn.get_contents();

	ASSERT_EQ(exrbm.size(), gotrbm.size());
	ASSERT_EQ(5, gotrbm.size());
	ASSERT_EQ(exrrbm.size(), gotrrbm.size());
	ASSERT_EQ(5, gotrrbm.size());
	ASSERT_EQ(exnobias.size(), gonobias.size());
	ASSERT_EQ(5, gonobias.size());

	ASSERT_EQ(exrbm[0], gotrbm[0]);
	ASSERT_EQ(exrbm[1], gotrbm[1]);
	ASSERT_EQ(exrbm[3], gotrbm[3]);

	ASSERT_EQ(exrrbm[0], gotrrbm[0]);
	ASSERT_EQ(exrrbm[1], gotrrbm[1]);
	ASSERT_EQ(exrrbm[3], gotrrbm[3]);

	ASSERT_EQ(exnobias[0], gonobias[0]);
	ASSERT_EQ(nullptr, exnobias[1]);
	ASSERT_EQ(nullptr, gonobias[1]);
	ASSERT_EQ(nullptr, exnobias[3]);
	ASSERT_EQ(nullptr, gonobias[3]);
}


#endif // DISABLE_RBM_TEST
