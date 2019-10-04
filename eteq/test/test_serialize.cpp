
#ifndef DISABLE_SERIALIZE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "diff/msg.hpp"

#include "exam/exam.hpp"

#include "dbg/stream/teq.hpp"

#include "pbm/save.hpp"
#include "pbm/load.hpp"

#include "tag/prop.hpp"

#include "eteq/serialize.hpp"
#include "eteq/eteq.hpp"


const std::string testdir = "models/test";


TEST(SERIALIZE, SaveGraph)
{
	std::string expect_pbfile = testdir + "/eteq_test.pbx";
	std::string got_pbfile = "got_eteq_test.pbx";
	cortenn::Graph graph;

	teq::Shape in_shape({10, 3});
	teq::Shape weight0_shape({9, 10});
	teq::Shape bias0_shape({9});
	teq::Shape weight1_shape({5, 9});
	teq::Shape bias1_shape({5});
	teq::Shape out_shape({5,3});

	eteq::NodeptrT<double> in = eteq::make_variable<double>(
		std::vector<double>(in_shape.n_elems()).data(),
		in_shape, "in");
	eteq::NodeptrT<double> weight0 = eteq::make_variable<double>(
		std::vector<double>(weight0_shape.n_elems()).data(),
		weight0_shape, "weight0");
	eteq::NodeptrT<double> bias0 = eteq::make_variable<double>(
		std::vector<double>(bias0_shape.n_elems()).data(),
		bias0_shape, "bias0");
	eteq::NodeptrT<double> weight1 = eteq::make_variable<double>(
		std::vector<double>(weight1_shape.n_elems()).data(),
		weight1_shape, "weight1");
	eteq::NodeptrT<double> bias1 = eteq::make_variable<double>(
		std::vector<double>(bias1_shape.n_elems()).data(),
		bias1_shape, "bias1");
	eteq::NodeptrT<double> out = eteq::make_variable<double>(
		std::vector<double>(out_shape.n_elems()).data(),
		out_shape, "out");

	auto& preg = tag::get_property_reg();
	preg.property_tag(in->get_tensor(), "training_in");
	preg.property_tag(weight0->get_tensor(), "storage_weight0");
	preg.property_tag(bias0->get_tensor(), "storage_bias0");
	preg.property_tag(weight1->get_tensor(), "storage_weight1");
	preg.property_tag(bias1->get_tensor(), "storage_bias1");
	preg.property_tag(out->get_tensor(), "training_out");

	auto layer0 = tenncor::matmul(in, weight0) + tenncor::extend(bias0, 1, {3});
	auto sig0 = 1. / (
		eteq::make_constant_scalar<double>(1, teq::Shape({9, 3})) +
		tenncor::exp(-layer0));

	auto layer1 = tenncor::matmul(sig0, weight1) + tenncor::extend(bias1, 1, {3});
	auto sig1 = 1. / (
		eteq::make_constant_scalar<double>(1, teq::Shape({5, 3})) +
		tenncor::exp(-layer1));

	auto err = tenncor::pow(out - sig1, 2.);

	auto dw0 = eteq::derive(err, weight0);
	auto db0 = eteq::derive(err, bias0);
	auto dw1 = eteq::derive(err, weight1);
	auto db1 = eteq::derive(err, bias1);

	preg.property_tag(dw0->get_tensor(), "derivative_dw0");
	preg.property_tag(db0->get_tensor(), "derivative_db0");
	preg.property_tag(dw1->get_tensor(), "derivative_dw1");
	preg.property_tag(db1->get_tensor(), "derivative_db1");

	pbm::GraphSaver<eteq::EADSaver> saver;
	dw0->get_tensor()->accept(saver);
	db0->get_tensor()->accept(saver);
	dw1->get_tensor()->accept(saver);
	db1->get_tensor()->accept(saver);

	saver.save(graph);

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
		std::fstream inputstr(testdir + "/eteq_test.pbx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(in.ParseFromIstream(&inputstr));
	}

	teq::TensptrSetT out;
	pbm::load_graph<eteq::EADLoader>(out, in);
	EXPECT_EQ(4, out.size());

	auto& reg = tag::get_reg();
	tag::Query q;

	std::vector<std::string> root_props;
	std::unordered_map<std::string,teq::TensptrT> propdtens;
	for (auto tens : out)
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
		std::ifstream expectstr(testdir + "/eteq_test.txt");
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
