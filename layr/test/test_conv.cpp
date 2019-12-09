
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
	EXPECT_STREQ(layr::conv_weight_key.c_str(), gotconv[0]->to_string().c_str());
	EXPECT_STREQ(layr::conv_bias_key.c_str(), gotconv[1]->to_string().c_str());
	EXPECT_TENSDATA(exconv[0].get(), gotconv[0].get(), PybindT);
	EXPECT_TENSDATA(exconv[1].get(), gotconv[1].get(), PybindT);

	ASSERT_NE(exrconv[0], gotrconv[0]);
	ASSERT_NE(exrconv[1], gotrconv[1]);
	EXPECT_STREQ(layr::conv_weight_key.c_str(), gotrconv[0]->to_string().c_str());
	EXPECT_STREQ(layr::conv_bias_key.c_str(), gotrconv[1]->to_string().c_str());
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


TEST(CONV, Connection)
{
	std::string label = "especially_convoluted";
	layr::Conv conv({6, 5}, 4, 3, label);

	auto x = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape({4, 10, 9, 2}), "x");
	auto y = conv.connect(eteq::to_link<PybindT>(x));

	EXPECT_GRAPHEQ(
		"(ADD[3\\6\\4\\2\\1\\1\\1\\1])\n"
		" `--(PERMUTE[3\\6\\4\\2\\1\\1\\1\\1])\n"
		" |   `--(CONV[1\\6\\4\\2\\3\\1\\1\\1])\n"
		" |       `--(PAD[4\\10\\9\\2\\5\\1\\1\\1])\n"
		" |       |   `--(variable:x[4\\10\\9\\2\\1\\1\\1\\1])\n"
		" |       `--(REVERSE[3\\4\\5\\6\\1\\1\\1\\1])\n"
		" |           `--(variable:weight[3\\4\\5\\6\\1\\1\\1\\1])\n"
		" `--(EXTEND[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"     `--(variable:bias[3\\1\\1\\1\\1\\1\\1\\1])",
		y->get_tensor());
}


TEST(CONV, Tagging)
{
	std::string label = "very_convoluted";
	layr::Conv conv({4, 5}, 6, 7, label);

	auto contents = conv.get_contents();
	// expect contents to be tagged
	ASSERT_EQ(2, contents.size());

	auto& reg = tag::get_reg();
	auto weight_tags = reg.get_tags(contents[0].get());
	auto bias_tags = reg.get_tags(contents[1].get());

	EXPECT_EQ(1, weight_tags.size());
	EXPECT_EQ(1, bias_tags.size());

	ASSERT_HAS(weight_tags, layr::conv_layer_key);
	ASSERT_HAS(bias_tags, layr::conv_layer_key);

	auto weight_labels = weight_tags[layr::conv_layer_key];
	auto bias_labels = bias_tags[layr::conv_layer_key];

	ASSERT_EQ(1, weight_labels.size());
	ASSERT_EQ(1, bias_labels.size());
	EXPECT_STREQ("very_convoluted::weight:0", weight_labels[0].c_str());
	EXPECT_STREQ("very_convoluted::bias:0", bias_labels[0].c_str());
}


#endif // DISABLE_CONV_TEST
