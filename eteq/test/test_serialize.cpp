
#ifndef DISABLE_SERIALIZE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "diff/msg.hpp"

#include "exam/exam.hpp"

#include "dbg/stream/teq.hpp"

#include "eteq/eteq.hpp"


const std::string testdir = "models/test";


TEST(SERIALIZE, SaveGraph)
{
	std::string expect_pbfile = testdir + "/eteq.onnx";
	std::string got_pbfile = "got_eteq.onnx";
	onnx::GraphProto graph;

	teq::Shape in_shape({10, 3});
	teq::Shape weight0_shape({9, 10});
	teq::Shape bias0_shape({9});
	teq::Shape weight1_shape({5, 9});
	teq::Shape bias1_shape({5});
	teq::Shape out_shape({5,3});

	eteq::LinkptrT<double> in = eteq::to_link<double>(
		eteq::make_variable<double>(
		std::vector<double>(in_shape.n_elems()).data(),
		in_shape, "in"));
	eteq::LinkptrT<double> weight0 = eteq::to_link<double>(
		eteq::make_variable<double>(
		std::vector<double>(weight0_shape.n_elems()).data(),
		weight0_shape, "weight0"));
	eteq::LinkptrT<double> bias0 = eteq::to_link<double>(
		eteq::make_variable<double>(
		std::vector<double>(bias0_shape.n_elems()).data(),
		bias0_shape, "bias0"));
	eteq::LinkptrT<double> weight1 = eteq::to_link<double>(
		eteq::make_variable<double>(
		std::vector<double>(weight1_shape.n_elems()).data(),
		weight1_shape, "weight1"));
	eteq::LinkptrT<double> bias1 = eteq::to_link<double>(
		eteq::make_variable<double>(
		std::vector<double>(bias1_shape.n_elems()).data(),
		bias1_shape, "bias1"));
	eteq::LinkptrT<double> out = eteq::to_link<double>(
		eteq::make_variable<double>(
		std::vector<double>(out_shape.n_elems()).data(),
		out_shape, "out"));

	auto layer0 = tenncor::matmul(in, weight0) + tenncor::extend(bias0, 1, {3});
	auto sig0 = 1. / (1. + tenncor::exp(-layer0));

	auto layer1 = tenncor::matmul(sig0, weight1) + tenncor::extend(bias1, 1, {3});
	auto sig1 = 1. / (1. + tenncor::exp(-layer1));

	auto err = tenncor::pow(out - sig1, 2.);

	auto dw0 = eteq::derive(err, weight0);
	auto db0 = eteq::derive(err, bias0);
	auto dw1 = eteq::derive(err, weight1);
	auto db1 = eteq::derive(err, bias1);

	eteq::save_graph(graph, {
		dw0->get_tensor(),
		db0->get_tensor(),
		dw1->get_tensor(),
		db1->get_tensor(),
	});
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

		onnx::GraphProto expect_graph;
		onnx::GraphProto got_graph;
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
	onnx::GraphProto in;
	{
		std::fstream inputstr(testdir + "/eteq.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(in.ParseFromIstream(&inputstr));
	}

	teq::TensptrsT out;
	eteq::load_graph(out, in);
	EXPECT_EQ(4, out.size());

	std::string expect;
	std::string got;
	std::string line;
	{
		std::ifstream expectstr(testdir + "/eteq.txt");
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
	for (auto t : out)
	{
		artist.print(gotstr, t);
	}

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
