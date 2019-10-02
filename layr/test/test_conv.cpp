
#ifndef DISABLE_CONV_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "layr/conv.hpp"


TEST(CONV, Copy)
{
	std::string label = "especially_convoluted";
	std::string rlabel = "kinda_onvoluted";
	layr::Conv conv({4, 5}, 6, 7, label);
	layr::Conv rconv({5, 6}, 7, 8, rlabel);

	auto exconv = conv.get_contents();
	auto exrconv = rconv.get_contents();

	layr::Conv cpyd(conv);
	layr::Conv cpyr(rconv);

	EXPECT_STREQ(label.c_str(), cpyd.get_label().c_str());
	EXPECT_STREQ(rlabel.c_str(), cpyr.get_label().c_str());

	EXPECT_EQ(6, cpyd.get_ninput());
	EXPECT_EQ(7, cpyr.get_ninput());

	EXPECT_EQ(7, cpyd.get_noutput());
	EXPECT_EQ(8, cpyr.get_noutput());

	auto gotconv = cpyd.get_contents();
	auto gotrconv = cpyr.get_contents();

	ASSERT_EQ(exconv.size(), gotconv.size());
	ASSERT_EQ(2, gotconv.size());
	ASSERT_EQ(exrconv.size(), gotrconv.size());
	ASSERT_EQ(2, gotrconv.size());

	ASSERT_NE(exconv[0], gotconv[0]);
	ASSERT_NE(exconv[1], gotconv[1]);
	EXPECT_STREQ("weight", gotconv[0]->to_string().c_str());
	EXPECT_STREQ("bias", gotconv[1]->to_string().c_str());
	EXPECT_TENSDATA(exconv[0].get(), gotconv[0].get(), PybindT);
	EXPECT_TENSDATA(exconv[1].get(), gotconv[1].get(), PybindT);

	ASSERT_NE(exrconv[0], gotrconv[0]);
	ASSERT_NE(exrconv[1], gotrconv[1]);
	EXPECT_STREQ("weight", gotrconv[0]->to_string().c_str());
	EXPECT_STREQ("bias", gotrconv[1]->to_string().c_str());
	EXPECT_TENSDATA(exrconv[0].get(), gotrconv[0].get(), PybindT);
	EXPECT_TENSDATA(exrconv[1].get(), gotrconv[1].get(), PybindT);
}


TEST(CONV, Clone)
{
	std::string label = "especially_convoluted";
	std::string rlabel = "kinda_onvoluted";
	layr::Conv conv({4, 5}, 6, 7, label);
	layr::Conv rconv({5, 6}, 7, 8, rlabel);

	layr::Conv* cpyd = conv.clone("ghi");
	layr::Conv* cpyr = rconv.clone("def");

	EXPECT_STREQ(("ghi" + label).c_str(), cpyd->get_label().c_str());
	EXPECT_STREQ(("def" + rlabel).c_str(), cpyr->get_label().c_str());

	delete cpyd;
	delete cpyr;
}


TEST(CONV, Move)
{
	std::string label = "especially_convoluted";
	std::string rlabel = "kinda_onvoluted";
	layr::Conv conv({4, 5}, 6, 7, label);
	layr::Conv rconv({5, 6}, 7, 8, rlabel);

	auto exconv = conv.get_contents();
	auto exrconv = rconv.get_contents();

	layr::Conv mvd(std::move(conv));
	layr::Conv mvr(std::move(rconv));

	EXPECT_STREQ("", conv.get_label().c_str());
	EXPECT_STREQ("", rconv.get_label().c_str());
	EXPECT_STREQ(label.c_str(), mvd.get_label().c_str());
	EXPECT_STREQ(rlabel.c_str(), mvr.get_label().c_str());

	EXPECT_EQ(6, mvd.get_ninput());
	EXPECT_EQ(7, mvr.get_ninput());

	EXPECT_EQ(7, mvd.get_noutput());
	EXPECT_EQ(8, mvr.get_noutput());

	auto gotconv = mvd.get_contents();
	auto gotrconv = mvr.get_contents();

	ASSERT_EQ(exconv.size(), gotconv.size());
	ASSERT_EQ(2, gotconv.size());
	ASSERT_EQ(exrconv.size(), gotrconv.size());
	ASSERT_EQ(2, gotrconv.size());

	ASSERT_EQ(exconv[0], gotconv[0]);
	ASSERT_EQ(exconv[1], gotconv[1]);

	ASSERT_EQ(exrconv[0], gotrconv[0]);
	ASSERT_EQ(exrconv[1], gotrconv[1]);
}


#endif // DISABLE_CONV_TEST
