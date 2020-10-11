
#ifndef DISABLE_SERIAL_SERIALIZE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "diff/diff.hpp"
#include "exam/exam.hpp"

#include "dbg/print/teq.hpp"

#include "internal/global/mock/mock.hpp"

#include "tenncor/serial/serial.hpp"


const std::string testdir = "models/test";


TEST(SERIALIZE, SaveGraph)
{
	std::string expect_pbfile = testdir + "/serial.onnx";
	std::string got_pbfile = "/tmp/serial.onnx";
	global::set_generator(std::make_shared<MockGenerator>());

	{
		onnx::ModelProto model;
		std::vector<teq::TensptrT> roots;
		onnx::TensIdT ids;

		// subtree one
		teq::Shape shape({3, 7});

		teq::TensptrT osrc(eteq::make_variable<float>(
			std::vector<float>(shape.n_elems()).data(), shape, "osrc"));
		teq::TensptrT osrc2(eteq::make_variable<double>(
			std::vector<double>(shape.n_elems()).data(), shape, "osrc2"));

		{
			teq::TensptrT src(eteq::make_variable<int32_t>(
				std::vector<int32_t>(shape.n_elems()).data(), shape, "src"));
			teq::TensptrT src2(eteq::make_constant_scalar<double>(23, shape));

			teq::TensptrT dest = eteq::make_functor(egen::SUB, {
				src2, eteq::make_functor(egen::POW, {
					eteq::make_functor(egen::DIV, {
						eteq::make_functor(egen::NEG, {osrc}),
						eteq::make_functor(egen::ADD, {
							eteq::make_functor(egen::SIN, {src}), src,
						}),
					}),
					osrc2,
				}),
			});
			roots.push_back(dest);
			ids.insert({dest.get(), "root1"});
		}

		// subtree two
		{
			teq::TensptrT src(eteq::Variable<float>::get(
				std::vector<float>(shape.n_elems()).data(), shape,
					"s2src", teq::PLACEHOLDER));
			teq::TensptrT src2(eteq::make_variable<int32_t>(
				std::vector<int32_t>(shape.n_elems()).data(), shape, "s2src2"));
			teq::TensptrT src3(eteq::make_variable<double>(
				std::vector<double>(shape.n_elems()).data(), shape, "s2src3"));

			teq::TensptrT dest = eteq::make_functor(egen::SUB, {
				src, eteq::make_functor(egen::MUL, {
					eteq::make_functor(egen::ABS, {src}),
					eteq::make_functor(egen::EXP, {src2}),
					eteq::make_functor(egen::NEG, {src3}),
				}),
			});
			roots.push_back(dest);
			ids.insert({dest.get(), "root2"});
		}

		serial::save_graph(*model.mutable_graph(), roots, ids);

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
		std::fstream inputstr(testdir + "/serial.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(in.ParseFromIstream(&inputstr));
	}

	onnx::TensptrIdT ids;
	auto out = serial::load_graph(ids, in.graph());
	EXPECT_EQ(2, out.size());

	std::string expect;
	std::string got;
	std::string line;
	std::ifstream expectstr(testdir + "/serial.txt");
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
	artist.cfg_.showshape_ = true;
	artist.cfg_.showtype_ = true;
	std::stringstream gotstr;

	ASSERT_HAS(ids.right, "root1");
	ASSERT_HAS(ids.right, "root2");
	auto root1 = ids.right.at("root1");
	auto root2 = ids.right.at("root2");
	ASSERT_NE(nullptr, root1);
	ASSERT_NE(nullptr, root2);
	artist.print(gotstr, root1);
	artist.print(gotstr, root2);

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


#endif // DISABLE_SERIAL_SERIALIZE_TEST
