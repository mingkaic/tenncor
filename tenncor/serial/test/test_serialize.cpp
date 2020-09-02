
#ifndef DISABLE_SERIALIZE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "diff/diff.hpp"
#include "exam/exam.hpp"

#include "dbg/print/teq.hpp"

#include "tenncor/serial/serial.hpp"
#include "tenncor/tenncor.hpp"


const std::string testdir = "models/test";


TEST(SERIALIZE, SaveGraph)
{
	std::string expect_pbfile = testdir + "/eteq.onnx";
	std::string got_pbfile = "got_eteq.onnx";
	onnx::ModelProto model;

	teq::Shape in_shape({10, 3});
	teq::Shape weight0_shape({9, 10});
	teq::Shape bias0_shape({9});
	teq::Shape weight1_shape({5, 9});
	teq::Shape bias1_shape({5});
	teq::Shape out_shape({5,3});

	eteq::ETensor in = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(in_shape.n_elems()).data(), in_shape, "in"));
	eteq::ETensor weight0 = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(weight0_shape.n_elems()).data(), weight0_shape, "weight0"));
	eteq::ETensor bias0 = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(bias0_shape.n_elems()).data(), bias0_shape, "bias0"));
	eteq::ETensor weight1 = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(weight1_shape.n_elems()).data(), weight1_shape, "weight1"));
	eteq::ETensor bias1 = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(bias1_shape.n_elems()).data(), bias1_shape, "bias1"));
	eteq::ETensor out = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(out_shape.n_elems()).data(), out_shape, "out"));

	auto layer0 = tenncor().matmul(in, weight0) + tenncor().extend(bias0, 1, {3});
	auto sig0 = 1. / ((double) 1. + tenncor().exp(-layer0));

	auto layer1 = tenncor().matmul(sig0, weight1) + tenncor().extend(bias1, 1, {3});
	auto sig1 = 1. / (1. + tenncor().exp(-layer1));

	auto err = tenncor().pow(out - sig1, 2.);

	auto dw0 = tcr::derive(err, {weight0})[0];
	auto db0 = tcr::derive(err, {bias0})[0];
	auto dw1 = tcr::derive(err, {weight1})[0];
	auto db1 = tcr::derive(err, {bias1})[0];

	onnx::TensIdT ids;
	ids.insert({dw0.get(), "dw0"});
	ids.insert({db0.get(), "db0"});
	ids.insert({dw1.get(), "dw1"});
	ids.insert({db1.get(), "db1"});
	tcr::save_model(model, {dw0, db0, dw1, db1}, ids);
	{
		std::fstream gotstr(got_pbfile,
			std::ios::out | std::ios::trunc | std::ios::binary);
		ASSERT_TRUE(gotstr.is_open());
		ASSERT_TRUE(model.SerializeToOstream(&gotstr));
	}

	{
		std::fstream expect_ifs(expect_pbfile, std::ios::in | std::ios::binary);
		std::fstream got_ifs(got_pbfile, std::ios::in | std::ios::binary);
		ASSERT_TRUE(expect_ifs.is_open());
		ASSERT_TRUE(got_ifs.is_open());

		onnx::ModelProto expect_model;
		onnx::ModelProto got_model;
		ASSERT_TRUE(expect_model.ParseFromIstream(&expect_ifs));
		ASSERT_TRUE(got_model.ParseFromIstream(&got_ifs));

		google::protobuf::util::MessageDifferencer differ;
		std::string report;
		differ.ReportDifferencesToString(&report);
		EXPECT_TRUE(differ.Compare(expect_model, got_model)) << report;
	}
}


TEST(SERIALIZE, SaveDependencies)
{
	std::string expect_pbfile = testdir + "/edeps.onnx";
	std::string got_pbfile = "got_edeps.onnx";
	onnx::ModelProto model;

	teq::Shape shape({10, 2});

	eteq::ETensor a = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(shape.n_elems()).data(), shape, "a"));
	eteq::ETensor b = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(shape.n_elems()).data(), shape, "b"));
	eteq::ETensor root = a * b;

	eteq::ETensor c = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(shape.n_elems()).data(), shape, "c"));
	eteq::ETensor dep = a + c;
	eteq::ETensor dep2 = a / c - b;

	tenncor().depends(root, {dep});
	tenncor().depends(root, {dep2});

	onnx::TensIdT ids;
	ids.insert({root.get(), "root"});
	tcr::save_model(model, {root}, ids);
	{
		std::fstream gotstr(got_pbfile,
			std::ios::out | std::ios::trunc | std::ios::binary);
		ASSERT_TRUE(gotstr.is_open());
		ASSERT_TRUE(model.SerializeToOstream(&gotstr));
	}

	{
		std::fstream expect_ifs(expect_pbfile, std::ios::in | std::ios::binary);
		std::fstream got_ifs(got_pbfile, std::ios::in | std::ios::binary);
		ASSERT_TRUE(expect_ifs.is_open());
		ASSERT_TRUE(got_ifs.is_open());

		onnx::ModelProto expect_model;
		onnx::ModelProto got_model;
		ASSERT_TRUE(expect_model.ParseFromIstream(&expect_ifs));
		ASSERT_TRUE(got_model.ParseFromIstream(&got_ifs));

		google::protobuf::util::MessageDifferencer differ;
		std::string report;
		differ.ReportDifferencesToString(&report);
		EXPECT_TRUE(differ.Compare(expect_model, got_model)) << report;
	}
}


TEST(SERIALIZE, LoadGraph)
{
	onnx::ModelProto in;
	{
		std::fstream inputstr(testdir + "/eteq.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(in.ParseFromIstream(&inputstr));
	}

	onnx::TensptrIdT ids;
	auto out = tcr::load_model(ids, in);
	EXPECT_EQ(4, out.size());

	ASSERT_HAS(ids.right, "dw0");
	ASSERT_HAS(ids.right, "db0");
	ASSERT_HAS(ids.right, "dw1");
	ASSERT_HAS(ids.right, "db1");
	auto dw0 = ids.right.at("dw0");
	auto db0 = ids.right.at("db0");
	auto dw1 = ids.right.at("dw1");
	auto db1 = ids.right.at("db1");
	ASSERT_NE(nullptr, dw0);
	ASSERT_NE(nullptr, db0);
	ASSERT_NE(nullptr, dw1);
	ASSERT_NE(nullptr, db1);

	std::string expect;
	std::string got;
	std::string line;
	{
		std::ifstream expectstr(testdir + "/eteq.txt");
		ASSERT_TRUE(expectstr.is_open());
		while (std::getline(expectstr, line))
		{
			fmts::strip(line, {' ', '\t', '\n', default_indent});
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

	std::ofstream os("eteq.json");
	artist.print(os, dw0);
	artist.print(os, db0);
	artist.print(os, dw1);
	artist.print(os, db1);

	while (std::getline(gotstr, line))
	{
		fmts::strip(line, {' ', '\t', '\n', default_indent});
		if (line.size() > 0)
		{
			got += line + '\n';
		}
	}

	EXPECT_STREQ(expect.c_str(), got.c_str());
}


#endif // DISABLE_SERIALIZE_TEST
