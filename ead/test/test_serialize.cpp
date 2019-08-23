
#ifndef DISABLE_SERIALIZE_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "dbg/stream/ade.hpp"

#include "pbm/save.hpp"
#include "pbm/load.hpp"

#include "ead/serialize.hpp"
#include "ead/ead.hpp"


const std::string testdir = "ead/data";


TEST(SERIALIZE, SaveGraph)
{
	std::string expect_pbfile = testdir + "/graph.pb";
	std::string got_pbfile = "got_graph.pb";
	cortenn::Graph graph;
	std::vector<ade::TensptrT> roots;

	pbm::PathedMapT labels;
	ade::Shape in_shape({10, 3});
	ade::Shape weight0_shape({9, 10});
	ade::Shape bias0_shape({9});
	ade::Shape weight1_shape({5, 9});
	ade::Shape bias1_shape({5});
	ade::Shape out_shape({5,3});

	ead::NodeptrT<double> in = ead::make_variable<double>(
		std::vector<double>(in_shape.n_elems()).data(), in_shape);
	ead::NodeptrT<double> weight0 = ead::make_variable<double>(
		std::vector<double>(weight0_shape.n_elems()).data(), weight0_shape);
	ead::NodeptrT<double> bias0 = ead::make_variable<double>(
		std::vector<double>(bias0_shape.n_elems()).data(), bias0_shape);
	ead::NodeptrT<double> weight1 = ead::make_variable<double>(
		std::vector<double>(weight1_shape.n_elems()).data(), weight1_shape);
	ead::NodeptrT<double> bias1 = ead::make_variable<double>(
		std::vector<double>(bias1_shape.n_elems()).data(), bias1_shape);
	ead::NodeptrT<double> out = ead::make_variable<double>(
		std::vector<double>(out_shape.n_elems()).data(), out_shape);

	labels[in->get_tensor()] = {"global", "training", "in"};
	labels[weight0->get_tensor()] = {"global", "storage", "weight0"};
	labels[bias0->get_tensor()] = {"global", "storage", "bias0"};
	labels[weight1->get_tensor()] = {"global", "storage", "weight1"};
	labels[bias1->get_tensor()] = {"global", "storage", "bias1"};
	labels[out->get_tensor()] = {"global", "training", "out"};

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

	labels[dw0->get_tensor()] = {"global", "dw0"};
	labels[db0->get_tensor()] = {"global", "db0"};
	labels[dw1->get_tensor()] = {"global", "dw1"};
	labels[db1->get_tensor()] = {"global", "db1"};

	pbm::GraphSaver<ead::EADSaver> saver;
	dw0->get_tensor()->accept(saver);
	db0->get_tensor()->accept(saver);
	dw1->get_tensor()->accept(saver);
	db1->get_tensor()->accept(saver);

	saver.save(graph, labels);

	{
		std::fstream gotstr(got_pbfile,
			std::ios::out | std::ios::trunc | std::ios::binary);
		ASSERT_TRUE(gotstr.is_open());
		ASSERT_TRUE(graph.SerializeToOstream(&gotstr));
	}

	std::fstream expect_ifs(expect_pbfile, std::ios::in | std::ios::binary);
	std::fstream got_ifs(got_pbfile, std::ios::in | std::ios::binary);
	ASSERT_TRUE(expect_ifs.is_open());
	ASSERT_TRUE(got_ifs.is_open());

	{
		cortenn::Graph expect_graph;
		cortenn::Graph got_graph;
		ASSERT_TRUE(expect_graph.ParseFromIstream(&expect_ifs));
		ASSERT_TRUE(got_graph.ParseFromIstream(&got_ifs));

		auto& expect_nodes = expect_graph.nodes();
		auto& got_nodes = got_graph.nodes();

		size_t nexpect_nodes = expect_nodes.size();
		ASSERT_EQ(nexpect_nodes, got_nodes.size());
		for (size_t i = 0; i < nexpect_nodes; ++i)
		{
			const cortenn::Node& expect_node = expect_nodes[i];
			const cortenn::Node& got_node = got_nodes[i];

			// check details
			if (expect_node.has_source())
			{
				const cortenn::Source& expect_source = expect_node.source();
				const cortenn::Source& got_source = got_node.source();

				std::string expect_shape = expect_source.shape();
				std::string got_shape = got_source.shape();
				EXPECT_STREQ(expect_shape.c_str(), got_shape.c_str());

				std::string expect_data = expect_source.data();
				std::string got_data = got_source.data();
				EXPECT_STREQ(expect_data.c_str(), got_data.c_str());

				std::string expect_type = expect_source.typelabel();
				std::string got_type = got_source.typelabel();
				EXPECT_STREQ(expect_type.c_str(), got_type.c_str());

				bool expect_const = expect_source.is_const();
				bool got_const = got_source.is_const();
				EXPECT_EQ(expect_const, got_const);
			}
			else
			{
				const cortenn::Functor& expect_func = expect_node.functor();
				const cortenn::Functor& got_func = got_node.functor();

				std::string expect_op = expect_func.opname();
				std::string got_op = got_func.opname();
				EXPECT_STREQ(expect_op.c_str(), got_op.c_str());

				auto& expect_args = expect_func.args();
				auto& got_args = got_func.args();

				size_t nexpect_args = expect_args.size();
				ASSERT_EQ(nexpect_args, got_args.size());
				for (size_t i = 0; i < nexpect_args; ++i)
				{
					auto& expect_arg = expect_args[i];
					auto& got_arg = got_args[i];

					auto& expect_shaper = expect_arg.shaper();
					auto& got_shaper = got_arg.shaper();

					auto& expect_coorder = expect_arg.coord();
					auto& got_coorder = got_arg.coord();

					EXPECT_EQ(expect_arg.idx(), got_arg.idx());
					EXPECT_ARREQ(expect_shaper, got_shaper);
					EXPECT_ARREQ(expect_coorder, got_coorder);
					EXPECT_EQ(expect_arg.fwd(), got_arg.fwd());
				}
			}

			// check tags
			auto& expect_tags = expect_node.tags();
			auto& got_tags = got_node.tags();
			EXPECT_EQ(expect_tags.size(), got_tags.size());
			for (auto& expectpair : expect_tags)
			{
				ASSERT_HAS(got_tags, expectpair.first);
				auto& expect_labels = expectpair.second.labels();
				auto& got_labels = got_tags.at(expectpair.first).labels();
				EXPECT_ARREQ(expect_labels, got_labels);
			}
		}
	}
}


TEST(SERIALIZE, LoadGraph)
{
	cortenn::Graph in;
	{
		std::fstream inputstr(testdir + "/graph.pb",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(in.ParseFromIstream(&inputstr));
	}

	pbm::GraphInfo out;
	pbm::load_graph<ead::EADLoader>(out, in);

	EXPECT_EQ(4, out.roots_.size());

	ASSERT_EQ(1, out.tens_.children_.size());
	ASSERT_EQ(0, out.tens_.tens_.size());
	ASSERT_HAS(out.tens_.children_, "global");

	auto global = out.tens_.children_["global"];
	ASSERT_EQ(2, global->children_.size());
	ASSERT_EQ(4, global->tens_.size());
	ASSERT_HAS(global->children_, "training");
	ASSERT_HAS(global->children_, "storage");
	ASSERT_HAS(global->tens_, "dw0");
	ASSERT_HAS(global->tens_, "db0");
	ASSERT_HAS(global->tens_, "dw1");
	ASSERT_HAS(global->tens_, "db1");

	auto training = global->children_["training"];
	auto storage = global->children_["storage"];
	ASSERT_EQ(0, training->children_.size());
	ASSERT_EQ(2, training->tens_.size());
	ASSERT_EQ(0, storage->children_.size());
	ASSERT_EQ(4, storage->tens_.size());
	ASSERT_HAS(training->tens_, "in");
	ASSERT_HAS(training->tens_, "out");
	ASSERT_HAS(storage->tens_, "weight0");
	ASSERT_HAS(storage->tens_, "bias0");
	ASSERT_HAS(storage->tens_, "weight1");
	ASSERT_HAS(storage->tens_, "bias1");

	auto dw0 = global->tens_["dw0"];
	auto db0 = global->tens_["db0"];
	auto dw1 = global->tens_["dw1"];
	auto db1 = global->tens_["db1"];

	ASSERT_ARRHAS(out.roots_, dw0);
	ASSERT_ARRHAS(out.roots_, db0);
	ASSERT_ARRHAS(out.roots_, dw1);
	ASSERT_ARRHAS(out.roots_, db1);

	ASSERT_NE(nullptr, dw0);
	ASSERT_NE(nullptr, db0);
	ASSERT_NE(nullptr, dw1);
	ASSERT_NE(nullptr, db1);

	std::string expect;
	std::string got;
	std::string line;
	std::ifstream expectstr(testdir + "/graph.txt");
	ASSERT_TRUE(expectstr.is_open());
	while (std::getline(expectstr, line))
	{
		fmts::trim(line);
		if (line.size() > 0)
		{
			expect += line + '\n';
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
