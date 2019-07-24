
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

	std::string expect;
	std::string got;
	// skip the first line (it contains timestamp)
	expect_ifs >> expect;
	got_ifs >> got;
	for (size_t lineno = 1; expect_ifs && got_ifs; ++lineno)
	{
		expect_ifs >> expect;
		got_ifs >> got;
		EXPECT_STREQ(expect.c_str(), got.c_str()) << "line number " << lineno;
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
