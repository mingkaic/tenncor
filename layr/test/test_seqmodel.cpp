
#ifndef DISABLE_SEQMODEL_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "layr/seqmodel.hpp"
#include "layr/dense.hpp"
#include "layr/rbm.hpp"
#include "layr/conv.hpp"


TEST(SEQMODEL, Copy)
{
	std::string dlabel = "especially_dense";
	std::string rlabel = "kinda_restrictive";
	std::string clabel = "maybe_convoluted";
	std::string label = "serious_series";
	layr::SequentialModel model(label);

	model.push_back(std::make_shared<layr::Dense>(4, 5,
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		nullptr,
		dlabel));
	model.push_back(std::make_shared<layr::RBM>(6, 4,
		layr::sigmoid(),
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		rlabel));
	model.push_back(std::make_shared<layr::Conv>(
		std::pair<teq::DimT,teq::DimT>{3, 4}, 6, 7, clabel));

	auto expectc = model.get_contents();

	layr::SequentialModel cpy(model);

	EXPECT_STREQ(label.c_str(), cpy.get_label().c_str());
	EXPECT_EQ(5, cpy.get_ninput());
	EXPECT_EQ(7, cpy.get_noutput());

	auto gotc = cpy.get_contents();

	ASSERT_EQ(expectc.size(), gotc.size());
	ASSERT_EQ(12, gotc.size());

	// dense
	ASSERT_NE(expectc[0], gotc[0]);
	ASSERT_NE(expectc[1], gotc[1]);
	EXPECT_STREQ(layr::dense_weight_key.c_str(), gotc[0]->to_string().c_str());
	EXPECT_STREQ(layr::dense_bias_key.c_str(), gotc[1]->to_string().c_str());
	EXPECT_TENSDATA(expectc[0].get(), gotc[0].get(), PybindT);
	EXPECT_TENSDATA(expectc[1].get(), gotc[1].get(), PybindT);

	// rbm
	ASSERT_NE(expectc[3], gotc[3]);
	ASSERT_NE(expectc[4], gotc[4]);
	ASSERT_NE(expectc[7], gotc[7]);
	EXPECT_STREQ(layr::dense_weight_key.c_str(), gotc[3]->to_string().c_str());
	EXPECT_STREQ(layr::dense_bias_key.c_str(), gotc[4]->to_string().c_str());
	EXPECT_STREQ(layr::dense_bias_key.c_str(), gotc[7]->to_string().c_str());
	EXPECT_TENSDATA(expectc[3].get(), gotc[3].get(), PybindT);
	EXPECT_TENSDATA(expectc[4].get(), gotc[4].get(), PybindT);
	EXPECT_TENSDATA(expectc[7].get(), gotc[7].get(), PybindT);

	// conv
	ASSERT_NE(expectc[10], gotc[10]);
	ASSERT_NE(expectc[11], gotc[11]);
	EXPECT_STREQ(layr::conv_weight_key.c_str(), gotc[10]->to_string().c_str());
	EXPECT_STREQ(layr::conv_bias_key.c_str(), gotc[11]->to_string().c_str());
	EXPECT_TENSDATA(expectc[10].get(), gotc[10].get(), PybindT);
	EXPECT_TENSDATA(expectc[11].get(), gotc[11].get(), PybindT);
}


TEST(SEQMODEL, Clone)
{
	std::string dlabel = "especially_dense";
	std::string rlabel = "kinda_restrictive";
	std::string clabel = "maybe_convoluted";
	std::string label = "serious_series";
	layr::SequentialModel model(label);

	model.push_back(std::make_shared<layr::Dense>(4, 5,
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		nullptr,
		dlabel));
	model.push_back(std::make_shared<layr::RBM>(6, 4,
		layr::sigmoid(),
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		rlabel));
	model.push_back(std::make_shared<layr::Conv>(
		std::pair<teq::DimT,teq::DimT>{3, 4}, 6, 7, clabel));

	auto expectc = model.get_contents();

	auto cpy = model.clone("consecutive_");

	EXPECT_STREQ(("consecutive_" + label).c_str(), cpy->get_label().c_str());
	EXPECT_EQ(5, cpy->get_ninput());
	EXPECT_EQ(7, cpy->get_noutput());

	delete cpy;
}


TEST(SEQMODEL, Move)
{
	std::string dlabel = "especially_dense";
	std::string rlabel = "kinda_restrictive";
	std::string clabel = "maybe_convoluted";
	std::string label = "serious_series";
	layr::SequentialModel model(label);

	model.push_back(std::make_shared<layr::Dense>(4, 5,
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		nullptr,
		dlabel));
	model.push_back(std::make_shared<layr::RBM>(6, 4,
		layr::sigmoid(),
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		rlabel));
	model.push_back(std::make_shared<layr::Conv>(
		std::pair<teq::DimT,teq::DimT>{3, 4}, 6, 7, clabel));

	auto expectc = model.get_contents();

	layr::SequentialModel mv(std::move(model));

	EXPECT_STREQ(label.c_str(), mv.get_label().c_str());
	EXPECT_EQ(5, mv.get_ninput());
	EXPECT_EQ(7, mv.get_noutput());

	auto gotc = mv.get_contents();

	ASSERT_EQ(expectc.size(), gotc.size());
	ASSERT_EQ(12, gotc.size());

	// dense
	ASSERT_EQ(expectc[0], gotc[0]);
	ASSERT_EQ(expectc[1], gotc[1]);

	// rbm
	ASSERT_EQ(expectc[3], gotc[3]);
	ASSERT_EQ(expectc[4], gotc[4]);
	ASSERT_EQ(expectc[7], gotc[7]);

	// conv
	ASSERT_EQ(expectc[10], gotc[10]);
	ASSERT_EQ(expectc[11], gotc[11]);
}


TEST(SEQMODEL, Tagging)
{
	std::string dlabel = "especially_dense";
	std::string rlabel = "kinda_restrictive";
	std::string clabel = "maybe_convoluted";
	std::string label = "serious_series";
	layr::SequentialModel model(label);

	model.push_back(std::make_shared<layr::Dense>(4, 5,
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		nullptr,
		dlabel));
	model.push_back(std::make_shared<layr::RBM>(6, 4,
		layr::sigmoid(),
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		rlabel));
	model.push_back(std::make_shared<layr::Conv>(
		std::pair<teq::DimT,teq::DimT>{3, 4}, 6, 7, clabel));

	auto contents = model.get_contents();
	// expect contents to be tagged
	ASSERT_EQ(12, contents.size());

	auto& reg = tag::get_reg();
	auto dense_weight_tags = reg.get_tags(contents[0].get());
	auto dense_bias_tags = reg.get_tags(contents[1].get());
	auto rbm_weight_tags = reg.get_tags(contents[3].get());
	auto rbm_hbias_tags = reg.get_tags(contents[4].get());
	auto rbm_perm_tags = reg.get_tags(contents[6].get());
	auto rbm_vbias_tags = reg.get_tags(contents[7].get());
	auto rbm_sig_tags = reg.get_tags(contents[9].get());
	auto conv_weight_tags = reg.get_tags(contents[10].get());
	auto conv_bias_tags = reg.get_tags(contents[11].get());

	EXPECT_EQ(2, dense_weight_tags.size());
	EXPECT_EQ(2, dense_bias_tags.size());
	EXPECT_EQ(3, rbm_weight_tags.size());
	EXPECT_EQ(3, rbm_hbias_tags.size());
	EXPECT_EQ(3, rbm_perm_tags.size());
	EXPECT_EQ(3, rbm_vbias_tags.size());
	EXPECT_EQ(4, rbm_sig_tags.size());
	EXPECT_EQ(2, conv_weight_tags.size());
	EXPECT_EQ(2, conv_bias_tags.size());

	ASSERT_HAS(dense_weight_tags, layr::seq_model_key);
	ASSERT_HAS(dense_bias_tags, layr::seq_model_key);
	ASSERT_HAS(rbm_weight_tags, layr::seq_model_key);
	ASSERT_HAS(rbm_hbias_tags, layr::seq_model_key);
	ASSERT_HAS(rbm_perm_tags, layr::seq_model_key);
	ASSERT_HAS(rbm_vbias_tags, layr::seq_model_key);
	ASSERT_HAS(rbm_sig_tags, layr::seq_model_key);
	ASSERT_HAS(conv_weight_tags, layr::seq_model_key);
	ASSERT_HAS(conv_bias_tags, layr::seq_model_key);

	auto seq_dense_weight_labels = dense_weight_tags[layr::seq_model_key];
	auto seq_dense_bias_labels = dense_bias_tags[layr::seq_model_key];
	auto seq_rbm_weight_labels = rbm_weight_tags[layr::seq_model_key];
	auto seq_rbm_hbias_labels = rbm_hbias_tags[layr::seq_model_key];
	auto seq_rbm_perm_labels = rbm_perm_tags[layr::seq_model_key];
	auto seq_rbm_vbias_labels = rbm_vbias_tags[layr::seq_model_key];
	auto seq_rbm_sig_labels = rbm_sig_tags[layr::seq_model_key];
	auto seq_conv_weight_labels = conv_weight_tags[layr::seq_model_key];
	auto seq_conv_bias_labels = conv_bias_tags[layr::seq_model_key];

	ASSERT_EQ(1, seq_dense_weight_labels.size());
	ASSERT_EQ(1, seq_dense_bias_labels.size());
	ASSERT_EQ(1, seq_rbm_weight_labels.size());
	ASSERT_EQ(1, seq_rbm_hbias_labels.size());
	ASSERT_EQ(1, seq_rbm_perm_labels.size());
	ASSERT_EQ(1, seq_rbm_vbias_labels.size());
	ASSERT_EQ(1, seq_rbm_sig_labels.size());
	ASSERT_EQ(1, seq_conv_weight_labels.size());
	ASSERT_EQ(1, seq_conv_bias_labels.size());
	EXPECT_STREQ("serious_series:layer_dense:especially_dense:0", seq_dense_weight_labels[0].c_str());
	EXPECT_STREQ("serious_series:layer_dense:especially_dense:0", seq_dense_bias_labels[0].c_str());
	EXPECT_STREQ("serious_series:layer_rbm:kinda_restrictive:1", seq_rbm_weight_labels[0].c_str());
	EXPECT_STREQ("serious_series:layer_rbm:kinda_restrictive:1", seq_rbm_hbias_labels[0].c_str());
	EXPECT_STREQ("serious_series:layer_rbm:kinda_restrictive:1", seq_rbm_perm_labels[0].c_str());
	EXPECT_STREQ("serious_series:layer_rbm:kinda_restrictive:1", seq_rbm_vbias_labels[0].c_str());
	EXPECT_STREQ("serious_series:layer_rbm:kinda_restrictive:1", seq_rbm_sig_labels[0].c_str());
	EXPECT_STREQ("serious_series:layer_conv:maybe_convoluted:2", seq_conv_weight_labels[0].c_str());
	EXPECT_STREQ("serious_series:layer_conv:maybe_convoluted:2", seq_conv_bias_labels[0].c_str());

	ASSERT_HAS(dense_weight_tags, layr::dense_layer_key);
	ASSERT_HAS(dense_bias_tags, layr::dense_layer_key);

	auto dense_weight_labels = dense_weight_tags[layr::dense_layer_key];
	auto dense_bias_labels = dense_bias_tags[layr::dense_layer_key];
	ASSERT_EQ(1, dense_weight_labels.size());
	ASSERT_EQ(1, dense_bias_labels.size());
	EXPECT_STREQ("especially_dense::weight:0", dense_weight_labels[0].c_str());
	EXPECT_STREQ("especially_dense::bias:0", dense_bias_labels[0].c_str());

	ASSERT_HAS(rbm_weight_tags, layr::rbm_layer_key);
	ASSERT_HAS(rbm_hbias_tags, layr::rbm_layer_key);
	ASSERT_HAS(rbm_perm_tags, layr::rbm_layer_key);
	ASSERT_HAS(rbm_vbias_tags, layr::rbm_layer_key);
	ASSERT_HAS(rbm_sig_tags, layr::rbm_layer_key);
	ASSERT_HAS(rbm_weight_tags, layr::dense_layer_key);
	ASSERT_HAS(rbm_hbias_tags, layr::dense_layer_key);
	ASSERT_HAS(rbm_perm_tags, layr::dense_layer_key);
	ASSERT_HAS(rbm_vbias_tags, layr::dense_layer_key);
	ASSERT_HAS(rbm_sig_tags, layr::sigmoid_layer_key);
	ASSERT_HAS(rbm_sig_tags, tag::props_key);

	auto rbm_weight_labels = rbm_weight_tags[layr::rbm_layer_key];
	auto rbm_hbias_labels = rbm_hbias_tags[layr::rbm_layer_key];
	auto rbm_perm_labels = rbm_perm_tags[layr::rbm_layer_key];
	auto rbm_vbias_labels = rbm_vbias_tags[layr::rbm_layer_key];
	auto rbm_sig_labels = rbm_sig_tags[layr::rbm_layer_key];
	ASSERT_EQ(1, rbm_weight_labels.size());
	ASSERT_EQ(1, rbm_hbias_labels.size());
	ASSERT_EQ(1, rbm_perm_labels.size());
	ASSERT_EQ(1, rbm_vbias_labels.size());
	ASSERT_EQ(1, rbm_sig_labels.size());
	EXPECT_STREQ("kinda_restrictive:layer_dense:hidden:0", rbm_weight_labels[0].c_str());
	EXPECT_STREQ("kinda_restrictive:layer_dense:hidden:0", rbm_hbias_labels[0].c_str());
	EXPECT_STREQ("kinda_restrictive:layer_dense:visible:1", rbm_perm_labels[0].c_str());
	EXPECT_STREQ("kinda_restrictive:layer_dense:visible:1", rbm_vbias_labels[0].c_str());
	EXPECT_STREQ("kinda_restrictive:layer_sigmoid::2", rbm_sig_labels[0].c_str());

	auto rbm_dense_weight_labels = rbm_weight_tags[layr::dense_layer_key];
	auto rbm_dense_hbias_labels = rbm_hbias_tags[layr::dense_layer_key];
	auto rbm_dense_perm_labels = rbm_perm_tags[layr::dense_layer_key];
	auto rbm_dense_vbias_labels = rbm_vbias_tags[layr::dense_layer_key];
	auto rbm_dense_sig_labels = rbm_sig_tags[layr::sigmoid_layer_key];
	auto sig_properties = rbm_sig_tags[tag::props_key];
	ASSERT_EQ(1, rbm_dense_weight_labels.size());
	ASSERT_EQ(1, rbm_dense_hbias_labels.size());
	ASSERT_EQ(1, rbm_dense_perm_labels.size());
	ASSERT_EQ(1, rbm_dense_vbias_labels.size());
	ASSERT_EQ(1, rbm_dense_sig_labels.size());
	ASSERT_EQ(1, sig_properties.size());
	EXPECT_STREQ("hidden::weight:0", rbm_dense_weight_labels[0].c_str());
	EXPECT_STREQ("hidden::bias:0", rbm_dense_hbias_labels[0].c_str());
	EXPECT_STREQ("visible::weight:0", rbm_dense_perm_labels[0].c_str());
	EXPECT_STREQ("visible::bias:0", rbm_dense_vbias_labels[0].c_str());
	EXPECT_STREQ("::uparam:0", rbm_dense_sig_labels[0].c_str());
	EXPECT_STREQ(tag::immutable_tag.c_str(), sig_properties[0].c_str());

	ASSERT_HAS(conv_weight_tags, layr::conv_layer_key);
	ASSERT_HAS(conv_bias_tags, layr::conv_layer_key);

	auto conv_weight_labels = conv_weight_tags[layr::conv_layer_key];
	auto conv_bias_labels = conv_bias_tags[layr::conv_layer_key];
	ASSERT_EQ(1, conv_weight_labels.size());
	ASSERT_EQ(1, conv_bias_labels.size());
	EXPECT_STREQ("maybe_convoluted::weight:0", conv_weight_labels[0].c_str());
	EXPECT_STREQ("maybe_convoluted::bias:0", conv_bias_labels[0].c_str());
}


#endif // DISABLE_SEQMODEL_TEST
