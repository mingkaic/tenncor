
#ifndef DISABLE_SERIALIZE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "diff/msg.hpp"

#include "exam/exam.hpp"

#include "dbg/stream/ade.hpp"

#include "pbm/save.hpp"
#include "pbm/load.hpp"

#include "tag/prop.hpp"

#include "ead/serialize.hpp"
#include "ead/ead.hpp"


const std::string testdir = "models/test";


TEST(SERIALIZE, SaveGraph)
{
	std::string expect_pbfile = testdir + "/ead_test.pbx";
	std::string got_pbfile = "got_ead_test.pbx";
	cortenn::Graph graph;

	ade::Shape in_shape({10, 3});
	ade::Shape weight0_shape({9, 10});
	ade::Shape bias0_shape({9});
	ade::Shape weight1_shape({5, 9});
	ade::Shape bias1_shape({5});
	ade::Shape out_shape({5,3});

	ead::NodeptrT<double> in = ead::make_variable<double>(
		std::vector<double>(in_shape.n_elems()).data(),
		in_shape, "in");
	ead::NodeptrT<double> weight0 = ead::make_variable<double>(
		std::vector<double>(weight0_shape.n_elems()).data(),
		weight0_shape, "weight0");
	ead::NodeptrT<double> bias0 = ead::make_variable<double>(
		std::vector<double>(bias0_shape.n_elems()).data(),
		bias0_shape, "bias0");
	ead::NodeptrT<double> weight1 = ead::make_variable<double>(
		std::vector<double>(weight1_shape.n_elems()).data(),
		weight1_shape, "weight1");
	ead::NodeptrT<double> bias1 = ead::make_variable<double>(
		std::vector<double>(bias1_shape.n_elems()).data(),
		bias1_shape, "bias1");
	ead::NodeptrT<double> out = ead::make_variable<double>(
		std::vector<double>(out_shape.n_elems()).data(),
		out_shape, "out");

	auto& preg = tag::get_property_reg();
	preg.property_tag(in->get_tensor(), "training_in");
	preg.property_tag(weight0->get_tensor(), "storage_weight0");
	preg.property_tag(bias0->get_tensor(), "storage_bias0");
	preg.property_tag(weight1->get_tensor(), "storage_weight1");
	preg.property_tag(bias1->get_tensor(), "storage_bias1");
	preg.property_tag(out->get_tensor(), "training_out");

	auto layer0 = tenncor::add(tenncor::matmul(in, weight0), tenncor::extend(bias0, 1, {3}));
	auto sig0 = tenncor::div(ead::make_constant_scalar<double>(1, ade::Shape({9, 3})),
		tenncor::add(ead::make_constant_scalar<double>(1, ade::Shape({9, 3})),
			tenncor::exp(tenncor::neg(layer0))));

	auto layer1 = tenncor::add(tenncor::matmul(sig0, weight1), tenncor::extend(bias1, 1, {3}));
	auto sig1 = tenncor::div(ead::make_constant_scalar<double>(1, ade::Shape({5, 3})),
		tenncor::add(ead::make_constant_scalar<double>(1, ade::Shape({5, 3})),
			tenncor::exp(tenncor::neg(layer1))));

	auto err = tenncor::pow(tenncor::sub(out, sig1), ead::make_constant_scalar<double>(2, out_shape));

	auto dw0 = ead::derive(err, weight0);
	auto db0 = ead::derive(err, bias0);
	auto dw1 = ead::derive(err, weight1);
	auto db1 = ead::derive(err, bias1);

	preg.property_tag(dw0->get_tensor(), "derivative_dw0");
	preg.property_tag(db0->get_tensor(), "derivative_db0");
	preg.property_tag(dw1->get_tensor(), "derivative_dw1");
	preg.property_tag(db1->get_tensor(), "derivative_db1");

	pbm::GraphSaver<ead::EADSaver> saver;
	dw0->get_tensor()->accept(saver);
	db0->get_tensor()->accept(saver);
	dw1->get_tensor()->accept(saver);
	db1->get_tensor()->accept(saver);

	pbm::PathedMapT labels; // todo: deprecate labels
	saver.save(graph, labels);

	{
		std::fstream gotstr(got_pbfile,
			std::ios::out | std::ios::trunc | std::ios::binary);
		ASSERT_TRUE(gotstr.is_open());
		ASSERT_TRUE(graph.SerializeToOstream(&gotstr));
	}

	{
		std::fstream expect_ifs(expect_pbfile, std::ios::in | std::ios::binary);
		std::fstream got_ifs(got_pbfile, std::ios::in | std::ios::binary);
		ASSERT_TRUE(expect_ifs.is_open());
		ASSERT_TRUE(got_ifs.is_open());

		cortenn::Graph expect_graph;
		cortenn::Graph got_graph;
		ASSERT_TRUE(expect_graph.ParseFromIstream(&expect_ifs));
		ASSERT_TRUE(got_graph.ParseFromIstream(&got_ifs));

		google::protobuf::util::MessageDifferencer differ;
		std::string report;
		differ.ReportDifferencesToString(&report);
		EXPECT_TRUE(differ.Compare(expect_graph, got_graph)) << report;
	}
}


TEST(SERIALIZE, LoadGraph)
{
	cortenn::Graph in;
	{
		std::fstream inputstr(testdir + "/ead_test.pbx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(in.ParseFromIstream(&inputstr));
	}

	pbm::GraphInfo out;
	pbm::load_graph<ead::EADLoader>(out, in);
	EXPECT_EQ(4, out.roots_.size());

	auto& reg = tag::get_reg();
	tag::Query q;

	std::vector<std::string> root_props;
	std::unordered_map<std::string,ade::TensptrT> propdtens;
	for (auto tens : out.roots_)
	{
		tens->accept(q);
		auto tags = reg.get_tags(tens.get());
		ASSERT_HAS(tags, tag::props_key);
		auto& props = tags[tag::props_key];
		for (auto& prop : props)
		{
			propdtens.emplace(prop, tens);
		}
		root_props.insert(root_props.end(), props.begin(), props.end());
	}
	EXPECT_ARRHAS(root_props, "derivative_dw0");
	EXPECT_ARRHAS(root_props, "derivative_dw1");
	EXPECT_ARRHAS(root_props, "derivative_db0");
	EXPECT_ARRHAS(root_props, "derivative_db1");

	ASSERT_HAS(q.labels_, tag::props_key);
	auto& props = q.labels_[tag::props_key];
	ASSERT_HAS(props, "training_in");
	ASSERT_HAS(props, "storage_weight0");
	ASSERT_HAS(props, "storage_bias0");
	ASSERT_HAS(props, "storage_weight1");
	ASSERT_HAS(props, "storage_bias1");
	ASSERT_HAS(props, "training_out");

	auto& tins = props["training_in"];
	auto& w0s = props["storage_weight0"];
	auto& b0s = props["storage_bias0"];
	auto& w1s = props["storage_weight1"];
	auto& b1s = props["storage_bias1"];
	auto& touts = props["training_out"];

	ASSERT_EQ(1, tins.size());
	ASSERT_EQ(1, w0s.size());
	ASSERT_EQ(1, b0s.size());
	ASSERT_EQ(1, w1s.size());
	ASSERT_EQ(1, b1s.size());
	ASSERT_EQ(1, touts.size());

	ASSERT_HAS(propdtens, "derivative_dw0");
	ASSERT_HAS(propdtens, "derivative_db0");
	ASSERT_HAS(propdtens, "derivative_dw1");
	ASSERT_HAS(propdtens, "derivative_db1");
	auto dw0 = propdtens["derivative_dw0"];
	auto db0 = propdtens["derivative_db0"];
	auto dw1 = propdtens["derivative_dw1"];
	auto db1 = propdtens["derivative_db1"];

	ASSERT_NE(nullptr, dw0);
	ASSERT_NE(nullptr, db0);
	ASSERT_NE(nullptr, dw1);
	ASSERT_NE(nullptr, db1);

	std::string expect;
	std::string got;
	std::string line;
	{
		std::ifstream expectstr(testdir + "/ead_test.txt");
		ASSERT_TRUE(expectstr.is_open());
		while (std::getline(expectstr, line))
		{
			fmts::trim(line);
			if (line.size() > 0)
			{
				expect += line + '\n';
			}
		}
	}

	PrettyEquation artist;
	std::stringstream gotstr;
	artist.print(gotstr, dw0);
	artist.print(gotstr, db0);
	artist.print(gotstr, dw1);
	artist.print(gotstr, db1);

	while (std::getline(gotstr, line))
	{
		fmts::trim(line);
		if (line.size() > 0)
		{
			got += line + '\n';
		}
	}

	EXPECT_STREQ(expect.c_str(), got.c_str());
}


#endif // DISABLE_SERIALIZE_TEST
