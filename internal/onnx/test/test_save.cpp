
#ifndef DISABLE_ONNX_SAVE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/leaf.hpp"
#include "internal/teq/mock/functor.hpp"

#include "internal/onnx/save.hpp"


const std::string testdir = "models/test";


struct MockMarshFuncs final : public onnx::iMarshFuncs
{
	size_t get_typecode (const teq::iTensor& tens) const override { return 0; }

	void marsh_leaf (onnx::TensorProto& out, const teq::iLeaf& leaf) const override {}
};


TEST(SAVE, BadMarshal)
{
	onnx::AttributeProto proto;
	teq::CTensMapT<std::string> tensid;
	onnx::OnnxAttrMarshaler marshal(&proto, tensid);

	marsh::ObjTuple tup;
	EXPECT_FATAL(tup.accept(marshal),
		"onnx does not support tuple attributes");

	marsh::Maps mm;
	EXPECT_FATAL(mm.accept(marshal),
		"onnx does not support map attributes");

	teq::LayerObj layer("banana_split",
		std::make_shared<MockLeaf>(std::vector<double>{3}, teq::Shape(), "m"));
	EXPECT_FATAL(layer.accept(marshal),
		"onnx does not support layer attributes");

	{
		auto nllog = new tutil::NoSupportLogger();
		global::set_logger(nllog);
		tup.accept(marshal);
		EXPECT_FALSE(nllog->called_);
	}
	{
		auto nllog = new tutil::NoSupportLogger();
		global::set_logger(nllog);
		mm.accept(marshal);
		EXPECT_FALSE(nllog->called_);
	}
	{
		auto nllog = new tutil::NoSupportLogger();
		global::set_logger(nllog);
		layer.accept(marshal);
		EXPECT_FALSE(nllog->called_);
	}

	global::set_logger(new exam::TestLogger());
}


TEST(SAVE, SimpleGraph)
{
	std::string expect_pbfile = testdir + "/simple_onnx.onnx";
	std::string got_pbfile = "got_simple_onnx.onnx";

	{
		onnx::ModelProto model;
		std::vector<teq::TensptrT> roots;
		onnx::TensIdT ids;

		// subtree one
		teq::Shape shape({3, 7});
		teq::Shape shape2({7, 3});
		teq::TensptrT osrc = std::make_shared<MockLeaf>(
			std::vector<double>{}, shape, "osrc");
		teq::TensptrT osrc2 = std::make_shared<MockLeaf>(
			std::vector<double>{}, shape2, "osrc2");

		{
			teq::Shape shape3({3, 1, 7});
			teq::TensptrT src = std::make_shared<MockLeaf>(
				std::vector<double>{}, shape, "src");
			teq::TensptrT src2 = std::make_shared<MockLeaf>(
				std::vector<double>{}, shape3, "src2");

			teq::TensptrT dest = std::make_shared<MockFunctor>(teq::TensptrsT{
				src2,
				std::make_shared<MockFunctor>(teq::TensptrsT{
					std::make_shared<MockFunctor>(teq::TensptrsT{
						std::make_shared<MockFunctor>(teq::TensptrsT{osrc}, teq::Opcode{"neg", 3}),
						std::make_shared<MockFunctor>(teq::TensptrsT{
							std::make_shared<MockFunctor>(teq::TensptrsT{src}, teq::Opcode{"sin", 5}),
							src,
						}, teq::Opcode{"+", 4}),
					}, teq::Opcode{"/", 2}),
					osrc2,
				}, teq::Opcode{"@", 1}),
			}, teq::Opcode{"-", 0});
			roots.push_back(dest);
			ids.insert({dest.get(), "root1"});
		}

		// subtree two
		{
			teq::Shape mshape({3, 3});
			teq::TensptrT src = std::make_shared<MockLeaf>(
				std::vector<double>{}, mshape, "s2src");
			teq::TensptrT src2 = std::make_shared<MockLeaf>(
				std::vector<double>{}, mshape, "s2src2");
			teq::TensptrT src3 = std::make_shared<MockLeaf>(
				std::vector<double>{}, mshape, "s2src3");

			teq::TensptrT dest = std::make_shared<MockFunctor>(teq::TensptrsT{
				src,
				std::make_shared<MockFunctor>(teq::TensptrsT{
					std::make_shared<MockFunctor>(teq::TensptrsT{src}, teq::Opcode{"abs", 7}),
					std::make_shared<MockFunctor>(teq::TensptrsT{src2}, teq::Opcode{"exp", 8}),
					std::make_shared<MockFunctor>(teq::TensptrsT{src3}, teq::Opcode{"neg", 3}),
				}, teq::Opcode{"*", 6}),
			}, teq::Opcode{"-", 0});
			roots.push_back(dest);
			ids.insert({dest.get(), "root2"});
		}

		MockMarshFuncs marsh;
		onnx::save_graph(*model.mutable_graph(), roots, marsh, ids);

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


struct MockGenerator : public global::iGenerator
{
	std::string get_str (void) const override
	{
		return "predictable";
	}

	int64_t unif_int (
		const int64_t& lower, const int64_t& upper) const override
	{
		return 0;
	}

	double unif_dec (
		const double& lower, const double& upper) const override
	{
		return 0;
	}

	double norm_dec (
		const double& mean, const double& stdev) const override
	{
		return 0;
	}

	global::GenF<std::string> get_strgen (void) const override
	{
		return []{ return ""; };
	}

	global::GenF<int64_t> unif_intgen (
		const int64_t& lower, const int64_t& upper) const override
	{
		return []{ return 0; };
	}

	global::GenF<double> unif_decgen (
		const double& lower, const double& upper) const override
	{
		return []{ return 0.; };
	}

	global::GenF<double> norm_decgen (
		const double& mean, const double& stdev) const override
	{
		return []{ return 0.; };
	}
};


TEST(SAVE, LayerGraph)
{
	std::string expect_pbfile = testdir + "/layer_onnx.onnx";
	std::string got_pbfile = "/tmp/layer_onnx.onnx";
	global::set_generator(std::make_shared<MockGenerator>());

	{
		onnx::ModelProto model;
		std::vector<teq::TensptrT> roots;
		onnx::TensIdT ids;

		// subtree one
		teq::Shape shape({3, 7});
		teq::Shape shape2({7, 3});
		teq::TensptrT osrc = std::make_shared<MockLeaf>(
			std::vector<double>{}, shape, "osrc");
		teq::TensptrT osrc2 = std::make_shared<MockLeaf>(
			std::vector<double>{}, shape2, "osrc2");

		static_cast<MockLeaf*>(osrc2.get())->usage_ = teq::PLACEHOLDER;

		{
			teq::Shape shape3({3, 1, 7});
			teq::TensptrT src = std::make_shared<MockLeaf>(
				std::vector<double>{}, shape, "src");
			teq::TensptrT src2 = std::make_shared<MockLeaf>(
				std::vector<double>{}, shape3, "src2");

			auto f0 = std::make_shared<MockFunctor>(teq::TensptrsT{
				src}, teq::Opcode{"sin", 5});
			f0->add_attr("num", std::make_unique<marsh::Number<size_t>>(54));
			f0->add_attr("array", std::make_unique<marsh::NumArray<double>>(
				std::vector<double>{3.3, 2.1, 7.6}));
			auto f1 = std::make_shared<MockFunctor>(teq::TensptrsT{
				f0, src,
			}, teq::Opcode{"+", 4});
			f1->add_attr(teq::layer_attr,
				std::make_unique<teq::LayerObj>("onion", f0));
			f1->add_attr("array", std::make_unique<marsh::NumArray<size_t>>(
				std::vector<size_t>{4, 6, 2}));
			f1->add_attr("num", std::make_unique<marsh::Number<double>>(4.3));
			ids.insert({f1.get(), "7"});

			teq::TensptrT dest = std::make_shared<MockFunctor>(teq::TensptrsT{
				src2, std::make_shared<MockFunctor>(teq::TensptrsT{
					std::make_shared<MockFunctor>(teq::TensptrsT{
						std::make_shared<MockFunctor>(teq::TensptrsT{
							osrc}, teq::Opcode{"neg", 3}), f1,
					}, teq::Opcode{"/", 2}),
					osrc2,
				}, teq::Opcode{"@", 1}),
			}, teq::Opcode{"-", 0});

			roots.push_back(dest);
			ids.insert({dest.get(), "root1"});
		}

		// subtree two
		{
			teq::Shape mshape({3, 3});
			teq::TensptrT src = std::make_shared<MockLeaf>(
				std::vector<double>{}, mshape, "s2src");
			teq::TensptrT src2 = std::make_shared<MockLeaf>(
				std::vector<double>{}, mshape, "s2src2");
			teq::TensptrT src3 = std::make_shared<MockLeaf>(
				std::vector<double>{}, mshape, "s2src3");

			auto f0 = std::make_shared<MockFunctor>(
				teq::TensptrsT{src}, teq::Opcode{"abs", 7});
			auto strs = std::make_unique<marsh::PtrArray<marsh::String>>();
			strs->contents_.insert(strs->contents_.end(),
				std::make_unique<marsh::String>("world"));
			strs->contents_.insert(strs->contents_.end(),
				std::make_unique<marsh::String>("food"));
			f0->add_attr("strs", std::move(strs));
			auto tensors = std::make_unique<marsh::PtrArray<teq::TensorObj>>();
			tensors->contents_.insert(tensors->contents_.end(),
				std::make_unique<teq::TensorObj>(osrc));
			tensors->contents_.insert(tensors->contents_.end(),
				std::make_unique<teq::TensorObj>(src3));
			f0->add_attr("tensors", std::move(tensors));
			auto f1 = std::make_shared<MockFunctor>(teq::TensptrsT{f0,
				std::make_shared<MockFunctor>(teq::TensptrsT{src2}, teq::Opcode{"exp", 8}),
				std::make_shared<MockFunctor>(teq::TensptrsT{src3}, teq::Opcode{"neg", 3}),
			}, teq::Opcode{"*", 6});
			f1->add_attr("str", std::make_unique<marsh::String>("hello"));
			auto destf = std::make_shared<MockFunctor>(teq::TensptrsT{
				src, f1,
			}, teq::Opcode{"-", 0});
			destf->add_attr("pineapple",
				std::make_unique<teq::TensorObj>(osrc));
			teq::TensptrT dest = destf;

			roots.push_back(dest);
			ids.insert({dest.get(), "root2"});
		}

		MockMarshFuncs marsh;
		onnx::save_graph(*model.mutable_graph(), roots, marsh, ids);

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


#endif // DISABLE_ONNX_SAVE_TEST
