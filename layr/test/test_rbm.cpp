
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
	EXPECT_STREQ(
		layr::dense_weight_key.c_str(),
		gotrbm[0]->to_string().c_str());
	EXPECT_STREQ(
		layr::dense_bias_key.c_str(),
		gotrbm[1]->to_string().c_str());
	EXPECT_STREQ(
		layr::dense_bias_key.c_str(),
		gotrbm[3]->to_string().c_str());
	EXPECT_TENSDATA(exrbm[0].get(), gotrbm[0].get(), PybindT);
	EXPECT_TENSDATA(exrbm[1].get(), gotrbm[1].get(), PybindT);
	EXPECT_TENSDATA(exrbm[3].get(), gotrbm[3].get(), PybindT);

	ASSERT_NE(exrrbm[0], gotrrbm[0]);
	ASSERT_NE(exrrbm[1], gotrrbm[1]);
	ASSERT_NE(exrrbm[3], gotrrbm[3]);
	EXPECT_STREQ(
		layr::dense_weight_key.c_str(),
		gotrrbm[0]->to_string().c_str());
	EXPECT_STREQ(
		layr::dense_bias_key.c_str(),
		gotrrbm[1]->to_string().c_str());
	EXPECT_STREQ(
		layr::dense_bias_key.c_str(),
		gotrrbm[3]->to_string().c_str());
	EXPECT_TENSDATA(exrrbm[0].get(), gotrrbm[0].get(), PybindT);
	EXPECT_TENSDATA(exrrbm[1].get(), gotrrbm[1].get(), PybindT);
	EXPECT_TENSDATA(exrrbm[3].get(), gotrrbm[3].get(), PybindT);

	ASSERT_NE(exnobias[0], gonobias[0]);
	ASSERT_EQ(exnobias[1], gonobias[1]);
	ASSERT_EQ(exnobias[3], gonobias[3]);
	ASSERT_EQ(nullptr, gonobias[1]);
	ASSERT_EQ(nullptr, gonobias[3]);
	EXPECT_STREQ(
		layr::dense_weight_key.c_str(),
		gonobias[0]->to_string().c_str());
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


TEST(RBM, Connection)
{
	std::string rlabel = "kinda_restrictive";
	std::string nb_label = "fake_news";
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

	auto x = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape({6, 2}), "x");
	auto x2 = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape({7, 2}), "x2");
	auto biasedy = rrbm.connect(x);
	auto y = nobias.connect(x2);

	EXPECT_GRAPHEQ(
		"(SIGMOID[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(ADD[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(MATMUL[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"     |   `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"     |   `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		"     `--(EXTEND[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"         `--(variable:bias[5\\1\\1\\1\\1\\1\\1\\1])",
		biasedy->get_tensor());

	EXPECT_GRAPHEQ(
		"(SIGMOID[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:x2[7\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])",
		y->get_tensor());
}


TEST(RBM, BackwardConnection)
{
	std::string rlabel = "kinda_restrictive";
	std::string nb_label = "fake_news";
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

	auto y = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape({5, 2}), "y");
	auto y2 = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape({6, 2}), "y2");
	auto biasedx = rrbm.backward_connect(y);
	auto x = nobias.backward_connect(y2);

	EXPECT_GRAPHEQ(
		"(SIGMOID[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(ADD[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"     |   `--(variable:y[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"     |   `--(PERMUTE[6\\5\\1\\1\\1\\1\\1\\1])\n"
		"     |       `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		"     `--(EXTEND[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"         `--(variable:bias[6\\1\\1\\1\\1\\1\\1\\1])",
		biasedx->get_tensor());

	EXPECT_GRAPHEQ(
		"(SIGMOID[7\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(MATMUL[7\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:y2[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(PERMUTE[7\\6\\1\\1\\1\\1\\1\\1])\n"
		"         `--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])",
		x->get_tensor());
}


TEST(RBM, Tagging)
{
	std::string label = "very_restrictive";
	layr::RBM rbm(5, 6,
		layr::sigmoid(),
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		label);

	auto contents = rbm.get_contents();
	// expect contents to be tagged
	ASSERT_EQ(5, contents.size());

	auto& reg = tag::get_reg();
	auto perm = contents[2];
	EXPECT_GRAPHEQ(
		"(PERMUTE[6\\5\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])",
		perm);
	auto weight_tags = reg.get_tags(contents[0].get());
	auto hbias_tags = reg.get_tags(contents[1].get());
	auto perm_tags = reg.get_tags(perm.get());
	auto vbias_tags = reg.get_tags(contents[3].get());
	auto sig_tags = reg.get_tags(contents[4].get());

	EXPECT_EQ(2, weight_tags.size());
	EXPECT_EQ(2, hbias_tags.size());
	EXPECT_EQ(2, perm_tags.size());
	EXPECT_EQ(2, vbias_tags.size());
	EXPECT_EQ(3, sig_tags.size());

	ASSERT_HAS(weight_tags, layr::rbm_layer_key);
	ASSERT_HAS(hbias_tags, layr::rbm_layer_key);
	ASSERT_HAS(perm_tags, layr::rbm_layer_key);
	ASSERT_HAS(vbias_tags, layr::rbm_layer_key);
	ASSERT_HAS(sig_tags, layr::rbm_layer_key);
	ASSERT_HAS(weight_tags, layr::dense_layer_key);
	ASSERT_HAS(hbias_tags, layr::dense_layer_key);
	ASSERT_HAS(perm_tags, layr::dense_layer_key);
	ASSERT_HAS(vbias_tags, layr::dense_layer_key);
	ASSERT_HAS(sig_tags, layr::sigmoid_layer_key);
	ASSERT_HAS(sig_tags, tag::props_key);

	auto rbm_weight_labels = weight_tags[layr::rbm_layer_key];
	auto rbm_hbias_labels = hbias_tags[layr::rbm_layer_key];
	auto rbm_perm_labels = perm_tags[layr::rbm_layer_key];
	auto rbm_vbias_labels = vbias_tags[layr::rbm_layer_key];
	auto rbm_sig_labels = sig_tags[layr::rbm_layer_key];

	ASSERT_EQ(1, rbm_weight_labels.size());
	ASSERT_EQ(1, rbm_hbias_labels.size());
	ASSERT_EQ(1, rbm_perm_labels.size());
	ASSERT_EQ(1, rbm_vbias_labels.size());
	ASSERT_EQ(1, rbm_sig_labels.size());
	EXPECT_STREQ("very_restrictive:layer_dense:hidden:0", rbm_weight_labels[0].c_str());
	EXPECT_STREQ("very_restrictive:layer_dense:hidden:0", rbm_hbias_labels[0].c_str());
	EXPECT_STREQ("very_restrictive:layer_dense:visible:1", rbm_perm_labels[0].c_str());
	EXPECT_STREQ("very_restrictive:layer_dense:visible:1", rbm_vbias_labels[0].c_str());
	EXPECT_STREQ("very_restrictive:layer_sigmoid::2", rbm_sig_labels[0].c_str());

	auto weight_labels = weight_tags[layr::dense_layer_key];
	auto hbias_labels = hbias_tags[layr::dense_layer_key];
	auto perm_labels = perm_tags[layr::dense_layer_key];
	auto vbias_labels = vbias_tags[layr::dense_layer_key];
	auto sig_labels = sig_tags[layr::sigmoid_layer_key];

	ASSERT_EQ(1, weight_labels.size());
	ASSERT_EQ(1, hbias_labels.size());
	ASSERT_EQ(1, perm_labels.size());
	ASSERT_EQ(1, vbias_labels.size());
	ASSERT_EQ(1, sig_labels.size());
	EXPECT_STREQ("hidden::weight:0", weight_labels[0].c_str());
	EXPECT_STREQ("hidden::bias:0", hbias_labels[0].c_str());
	EXPECT_STREQ("visible::weight:0", perm_labels[0].c_str());
	EXPECT_STREQ("visible::bias:0", vbias_labels[0].c_str());
	EXPECT_STREQ(":::0", sig_labels[0].c_str());

	auto sig_properties = sig_tags[tag::props_key];
	ASSERT_EQ(1, sig_properties.size());
	EXPECT_STREQ(tag::immutable_tag.c_str(), sig_properties[0].c_str());
}


#endif // DISABLE_RBM_TEST
