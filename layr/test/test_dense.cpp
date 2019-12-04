
#ifndef DISABLE_DENSE_TEST


#include "gtest/gtest.h"

#include "dbg/stream/teq_csv.hpp"

#include "testutil/tutil.hpp"

#include "layr/dense.hpp"


TEST(DENSE, Copy)
{
	std::string label = "especially_dense";
	std::string rlabel = "kinda_dense";
	std::string nb_label = "fake_news";
	layr::Dense dense(4, teq::Shape({5}),
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		nullptr,
		label);
	layr::Dense rdense(5, teq::Shape({6}),
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		nullptr,
		rlabel);
	layr::Dense nobias(6, teq::Shape({7}),
		layr::unif_xavier_init<PybindT>(3),
		layr::InitF<PybindT>(),
		nullptr,
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
	ASSERT_EQ(3, gotdense.size());
	ASSERT_EQ(exrdense.size(), gotrdense.size());
	ASSERT_EQ(3, gotrdense.size());
	ASSERT_EQ(exnobias.size(), gonobias.size());
	ASSERT_EQ(3, gonobias.size());

	ASSERT_NE(exdense[0], gotdense[0]);
	ASSERT_NE(exdense[1], gotdense[1]);
	EXPECT_EQ(exdense[2], gotdense[2]);
	EXPECT_EQ(nullptr, gotdense[2]);
	EXPECT_STREQ(
		layr::dense_weight_key.c_str(),
		gotdense[0]->to_string().c_str());
	EXPECT_STREQ(
		layr::dense_bias_key.c_str(),
		gotdense[1]->to_string().c_str());
	EXPECT_TENSDATA(exdense[0].get(), gotdense[0].get(), PybindT);
	EXPECT_TENSDATA(exdense[1].get(), gotdense[1].get(), PybindT);

	ASSERT_NE(exrdense[0], gotrdense[0]);
	ASSERT_NE(exrdense[1], gotrdense[1]);
	EXPECT_EQ(exrdense[2], gotrdense[2]);
	EXPECT_EQ(nullptr, gotrdense[2]);
	EXPECT_STREQ(
		layr::dense_weight_key.c_str(),
		gotrdense[0]->to_string().c_str());
	EXPECT_STREQ(
		layr::dense_bias_key.c_str(),
		gotrdense[1]->to_string().c_str());
	EXPECT_TENSDATA(exrdense[0].get(), gotrdense[0].get(), PybindT);
	EXPECT_TENSDATA(exrdense[1].get(), gotrdense[1].get(), PybindT);

	ASSERT_NE(exnobias[0], gonobias[0]);
	ASSERT_EQ(exnobias[1], gonobias[1]);
	EXPECT_EQ(gonobias[2], gonobias[2]);
	ASSERT_EQ(nullptr, gonobias[1]);
	EXPECT_EQ(nullptr, gonobias[2]);
	EXPECT_STREQ(
		layr::dense_weight_key.c_str(),
		gonobias[0]->to_string().c_str());
	EXPECT_TENSDATA(exnobias[0].get(), gonobias[0].get(), PybindT);
}


TEST(DENSE, Clone)
{
	std::string label = "especially_dense";
	std::string rlabel = "kinda_dense";
	std::string nb_label = "fake_news";
	layr::Dense dense(4, teq::Shape({5}),
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		nullptr,
		label);
	layr::Dense rdense(5, teq::Shape({6}),
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		nullptr,
		rlabel);
	layr::Dense nobias(6, teq::Shape({7}),
		layr::unif_xavier_init<PybindT>(3),
		layr::InitF<PybindT>(),
		nullptr,
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
	layr::Dense dense(4, teq::Shape({5}),
		layr::zero_init<PybindT>(),
		layr::zero_init<PybindT>(),
		nullptr,
		label);
	layr::Dense rdense(5, teq::Shape({6}),
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		nullptr,
		rlabel);
	layr::Dense nobias(6, teq::Shape({7}),
		layr::unif_xavier_init<PybindT>(3),
		layr::InitF<PybindT>(),
		nullptr,
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
	ASSERT_EQ(3, gotdense.size());
	ASSERT_EQ(exrdense.size(), gotrdense.size());
	ASSERT_EQ(3, gotrdense.size());
	ASSERT_EQ(exnobias.size(), gonobias.size());
	ASSERT_EQ(3, gonobias.size());

	ASSERT_EQ(exdense[0], gotdense[0]);
	ASSERT_EQ(exdense[1], gotdense[1]);
	EXPECT_EQ(exdense[2], gotdense[2]);

	ASSERT_EQ(exrdense[0], gotrdense[0]);
	ASSERT_EQ(exrdense[1], gotrdense[1]);
	EXPECT_EQ(exrdense[2], gotrdense[2]);

	ASSERT_EQ(exnobias[0], gonobias[0]);
	ASSERT_EQ(nullptr, exnobias[1]);
	ASSERT_EQ(nullptr, gonobias[1]);
	EXPECT_EQ(nullptr, exnobias[2]);
	EXPECT_EQ(nullptr, gonobias[2]);
}


TEST(DENSE, Connection)
{
	std::string rlabel = "kinda_dense";
	std::string nb_label = "fake_news";
	layr::Dense rdense(5, teq::Shape({6}),
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		nullptr,
		rlabel);
	layr::Dense nobias(6, teq::Shape({7}),
		layr::unif_xavier_init<PybindT>(3),
		layr::InitF<PybindT>(),
		nullptr,
		nb_label);

	auto x = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape({6, 2}), "x");
	auto x2 = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape({7, 2}), "x2");
	auto biasedy = rdense.connect(x);
	auto y = nobias.connect(x2);

	EXPECT_GRAPHEQ(
		"(ADD[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(MATMUL[5\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
		" `--(EXTEND[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:bias[5\\1\\1\\1\\1\\1\\1\\1])",
		biasedy->get_tensor());

	EXPECT_GRAPHEQ(
		"(MATMUL[6\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:x2[7\\2\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:weight[6\\7\\1\\1\\1\\1\\1\\1])",
		y->get_tensor());
}


TEST(DENSE, Tagging)
{
	std::string label = "very_dense";
	layr::Dense dense(5, teq::Shape({6}),
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		nullptr,
		label);

	auto contents = dense.get_contents();
	// expect contents to be tagged
	ASSERT_EQ(3, contents.size());

	auto& reg = tag::get_reg();
	auto weight_tags = reg.get_tags(contents[0].get());
	auto bias_tags = reg.get_tags(contents[1].get());

	EXPECT_EQ(1, weight_tags.size());
	EXPECT_EQ(1, bias_tags.size());

	ASSERT_HAS(weight_tags, layr::dense_layer_key);
	ASSERT_HAS(bias_tags, layr::dense_layer_key);

	auto weight_labels = weight_tags[layr::dense_layer_key];
	auto bias_labels = bias_tags[layr::dense_layer_key];

	ASSERT_EQ(1, weight_labels.size());
	ASSERT_EQ(1, bias_labels.size());
	EXPECT_STREQ("very_dense::weight:0", weight_labels[0].c_str());
	EXPECT_STREQ("very_dense::bias:0", bias_labels[0].c_str());
}


TEST(DENSE, ConnectionTagging)
{
	std::string label = "very_dense";
	layr::Dense dense(5, teq::Shape({6}),
		layr::unif_xavier_init<PybindT>(2),
		layr::unif_xavier_init<PybindT>(4),
		nullptr,
		label);

	auto x = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape({6, 2}), "x");
	auto y = dense.connect(x);

	auto contents = dense.get_contents();

	tag::Query q;
	y->get_tensor()->accept(q);

	EXPECT_EQ(3, q.labels_.size());
	ASSERT_HAS(q.labels_, layr::dense_layer_key);
	ASSERT_HAS(q.labels_, tag::groups_key);
	ASSERT_HAS(q.labels_, tag::props_key);
	auto denses = q.labels_[layr::dense_layer_key];
	auto groups = q.labels_[tag::groups_key];
	auto props = q.labels_[tag::props_key];

	EXPECT_EQ(3, denses.size());
	ASSERT_HAS(denses, "very_dense::weight:0");
	ASSERT_HAS(denses, "very_dense::bias:0");
	ASSERT_HAS(denses, "very_dense:::0");
	auto weights = denses["very_dense::weight:0"];
	auto biases = denses["very_dense::bias:0"];
	auto funcs = denses["very_dense:::0"];

	ASSERT_EQ(1, weights.size());
	ASSERT_EQ(1, biases.size());
	EXPECT_EQ(3, funcs.size());

	EXPECT_EQ(contents[0].get(), weights[0]);
	EXPECT_EQ(contents[1].get(), biases[0]);

	std::unordered_set<std::string> funcstrs;
	std::transform(funcs.begin(), funcs.end(),
		std::inserter(funcstrs, funcstrs.begin()),
		[](teq::iTensor* tens)
		{
			return tens->to_string();
		});
	EXPECT_HAS(funcstrs, "MATMUL");
	EXPECT_HAS(funcstrs, "ADD");
	EXPECT_HAS(funcstrs, "EXTEND");

	EXPECT_EQ(2, groups.size());
	ASSERT_HAS(groups, "fully_connect");
	ASSERT_HAS(groups, "sum");
	auto fcon_group = groups["fully_connect"];
	auto sum_group = groups["sum"];

	EXPECT_EQ(3, fcon_group.size());
	ASSERT_EQ(1, sum_group.size());

	std::unordered_set<std::string> fconstrs;
	std::transform(fcon_group.begin(), fcon_group.end(),
		std::inserter(fconstrs, fconstrs.begin()),
		[](teq::iTensor* tens)
		{
			return tens->to_string();
		});
	EXPECT_HAS(fconstrs, "MATMUL");
	EXPECT_HAS(fconstrs, "ADD");
	EXPECT_HAS(fconstrs, "EXTEND");

	EXPECT_STREQ("ADD", sum_group[0]->to_string().c_str());

	EXPECT_EQ(1, props.size());
	ASSERT_HAS(props, tag::commutative_tag);
	auto commies = props[tag::commutative_tag];

	ASSERT_EQ(1, commies.size());

	EXPECT_STREQ("ADD", commies[0]->to_string().c_str());
}


TEST(DENSE, Building)
{
	std::stringstream ss;
	std::string label = "very_dense";
	teq::DimT ninput = 6, noutput = 5;
	std::vector<PybindT> weight_data;
	std::vector<PybindT> bias_data;
	{
		// save
		layr::Dense dense(noutput, teq::Shape({ninput}),
			layr::unif_xavier_init<PybindT>(2),
			layr::unif_xavier_init<PybindT>(4),
			nullptr,
			label);

		auto contents = dense.get_contents();
		ASSERT_EQ(3, contents.size());
		auto weight = contents[0];
		auto bias = contents[1];
		auto params = contents[2];
		ASSERT_EQ(nullptr, params);
		PybindT* w = eteq::to_node<PybindT>(weight)->data();
		PybindT* b = eteq::to_node<PybindT>(bias)->data();
		weight_data = std::vector<PybindT>(w, w + weight->shape().n_elems());
		bias_data = std::vector<PybindT>(b, b + bias->shape().n_elems());

		auto x = eteq::make_variable_scalar<PybindT>(
			0, teq::Shape({6, 2}), "x");
		auto y = dense.connect(x);

		layr::save_layer(ss, dense, {y->get_tensor()});
	}
	ASSERT_EQ(noutput * ninput, weight_data.size());
	ASSERT_EQ(noutput, bias_data.size());
	{
		// load
		teq::TensptrsT roots;
		auto dense = layr::load_layer(ss, roots,
			layr::dense_layer_key, label);

		// verify layer
		auto contents = dense->get_contents();
		ASSERT_EQ(3, contents.size());
		auto weight = contents[0];
		auto bias = contents[1];
		teq::Shape exwshape({noutput, ninput});
		teq::Shape exbshape({noutput});
		auto wshape = weight->shape();
		auto bshape = bias->shape();
		ASSERT_ARREQ(exwshape, wshape);
		ASSERT_ARREQ(exbshape, bshape);
		PybindT* w = eteq::to_node<PybindT>(weight)->data();
		PybindT* b = eteq::to_node<PybindT>(bias)->data();
		std::vector<PybindT> gotw(w, w + weight->shape().n_elems());
		std::vector<PybindT> gotb(b, b + bias->shape().n_elems());
		EXPECT_VECEQ(weight_data, gotw);
		EXPECT_VECEQ(bias_data, gotb);

		// verify root
		ASSERT_EQ(1, roots.size());
		EXPECT_GRAPHEQ(
			"(ADD[5\\2\\1\\1\\1\\1\\1\\1])\n"
			" `--(MATMUL[5\\2\\1\\1\\1\\1\\1\\1])\n"
			" |   `--(variable:x[6\\2\\1\\1\\1\\1\\1\\1])\n"
			" |   `--(variable:weight[5\\6\\1\\1\\1\\1\\1\\1])\n"
			" `--(EXTEND[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"     `--(variable:bias[5\\1\\1\\1\\1\\1\\1\\1])", roots[0]);
	}
}


#endif // DISABLE_DENSE_TEST
