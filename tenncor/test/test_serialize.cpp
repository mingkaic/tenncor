
#ifndef DISABLE_TENNCOR_SERIALIZE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "diff/diff.hpp"
#include "exam/exam.hpp"

#include "dbg/print/teq.hpp"

#include "internal/global/mock/mock.hpp"

#include "tenncor/serial/serial.hpp"
#include "tenncor/tenncor.hpp"


using ::testing::Invoke;


#ifdef CMAKE_SOURCE_DIR
const std::string testdir = std::string(CMAKE_SOURCE_DIR) + "models/test";
#else
const std::string testdir = "models/test";
#endif


static eteq::ETensorsT mock_model (global::CfgMapptrT ctx)
{
	teq::Shape in_shape({10, 3});
	teq::Shape weight0_shape({9, 10});
	teq::Shape bias0_shape({9});
	teq::Shape weight1_shape({5, 9});
	teq::Shape bias1_shape({5});
	teq::Shape out_shape({5,3});

	eteq::ETensor in = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(in_shape.n_elems()).data(), in_shape, "in", ctx));
	eteq::ETensor weight0 = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(weight0_shape.n_elems()).data(), weight0_shape, "weight0", ctx));
	eteq::ETensor bias0 = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(bias0_shape.n_elems()).data(), bias0_shape, "bias0", ctx));
	eteq::ETensor weight1 = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(weight1_shape.n_elems()).data(), weight1_shape, "weight1", ctx));
	eteq::ETensor bias1 = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(bias1_shape.n_elems()).data(), bias1_shape, "bias1", ctx));
	eteq::ETensor out = eteq::ETensor(eteq::make_variable<double>(
		std::vector<double>(out_shape.n_elems()).data(), out_shape, "out", ctx));

	auto api = TenncorAPI(ctx);

	auto layer0 = api.add(api.matmul(in, weight0), api.extend(bias0, 1, {3}));
	auto sig0 = api.div(1., api.add(1., api.exp(api.neg(layer0))));

	auto layer1 = api.add(api.matmul(sig0, weight1), api.extend(bias1, 1, {3}));
	auto sig1 = api.div(1., api.add(1., api.exp(api.neg(layer1))));

	auto err = api.pow(api.sub(out, sig1), 2.);

	auto dw0 = tcr::derive(err, {weight0})[0];
	auto db0 = tcr::derive(err, {bias0})[0];
	auto dw1 = tcr::derive(err, {weight1})[0];
	auto db1 = tcr::derive(err, {bias1})[0];
	return {dw0, db0, dw1, db1};
}


TEST(SERIALIZE, SaveGraph)
{
	auto gen = std::make_shared<MockGenerator>();
	global::set_generator(gen);

	std::string expect_pbfile = testdir + "/eteq.onnx";
#ifdef EXPORT_TESTDATA
	std::string got_pbfile = "/tmp/eteq.onnx";
#else
	std::string got_pbfile = "got_eteq.onnx";
#endif

	size_t counter = 0;
	auto incr_id = [&]{ return fmts::to_string(++counter); };

	EXPECT_CALL(*gen, get_str()).
		WillRepeatedly(Invoke(incr_id));

	onnx::ModelProto model;
	auto mockders = mock_model(global::context());

	ASSERT_EQ(4, mockders.size());
	auto dw0 = mockders[0];
	auto db0 = mockders[1];
	auto dw1 = mockders[2];
	auto db1 = mockders[3];

	onnx::TensptrIdT ids;
	ids.insert({dw0, "dw0"});
	ids.insert({db0, "db0"});
	ids.insert({dw1, "dw1"});
	ids.insert({db1, "db1"});
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
	global::set_generator(nullptr);
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


TEST(SERIALIZE, SaveContext)
{
	auto gen = std::make_shared<MockGenerator>();
	global::set_generator(gen);

	std::string expect_pbfile = testdir + "/eteq_ctx.onnx";
#ifdef EXPORT_TESTDATA
	std::string got_pbfile = "/tmp/eteq_ctx.onnx";
#else
	std::string got_pbfile = "got_eteq_ctx.onnx";
#endif

	size_t counter = 0;
	auto incr_id = [&]{ return fmts::to_string(++counter); };

	EXPECT_CALL(*gen, get_str()).
		WillRepeatedly(Invoke(incr_id));

	onnx::ModelProto model;

	global::CfgMapptrT ctx = std::make_shared<estd::ConfigMap<>>();
	auto mockders = mock_model(ctx);

	ASSERT_EQ(4, mockders.size());
	auto dw0 = mockders[0];
	auto db0 = mockders[1];
	auto dw1 = mockders[2];
	auto db1 = mockders[3];

	ASSERT_EQ(ctx, dw0.get_context());
	ASSERT_EQ(ctx, db0.get_context());
	ASSERT_EQ(ctx, dw1.get_context());
	ASSERT_EQ(ctx, db1.get_context());

	tcr::save_model(model, ctx);
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
	global::set_generator(nullptr);
}


TEST(SERIALIZE, LoadContext)
{
	onnx::ModelProto in;
	{
		std::fstream inputstr(testdir + "/eteq_ctx.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(in.ParseFromIstream(&inputstr));
	}

	global::CfgMapptrT ctx = std::make_shared<estd::ConfigMap<>>();
	auto mockders = mock_model(ctx);

	ASSERT_EQ(4, mockders.size());
	auto dw0 = mockders[0];
	auto db0 = mockders[1];
	auto dw1 = mockders[2];
	auto db1 = mockders[3];

	auto out = tcr::load_model(ctx, in);
	EXPECT_EQ(4, out.size());

	teq::TensSetT outset;
	std::transform(out.begin(), out.end(),
		std::inserter(outset, outset.end()),
		[](eteq::ETensor& etens){ return etens.get(); });

	ASSERT_HAS(outset, dw0.get());
	ASSERT_HAS(outset, db0.get());
	ASSERT_HAS(outset, dw1.get());
	ASSERT_HAS(outset, db1.get());

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

#ifdef EXPORT_TESTDATA
	std::ofstream os("/tmp/eteq.json");
#else
	std::ofstream os("got_eteq.json");
#endif
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
#ifdef EXPORT_TESTDATA
	std::ofstream extract("/tmp/eteq.txt");
	extract << got;
#endif
}


#endif // DISABLE_TENNCOR_SERIALIZE_TEST
