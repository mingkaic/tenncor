
#ifndef DISABLE_SAVE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "internal/teq/mock/leaf.hpp"
#include "internal/teq/mock/functor.hpp"

#include "internal/onnx/save.hpp"


const std::string testdir = "models/test";


struct MockMarshFuncs final : public onnx::iMarshFuncs
{
	size_t get_typecode (const teq::iTensor& tens) const override { return 0; }

	void marsh_leaf (onnx::TensorProto& out, const teq::iLeaf& leaf) const override {}
};


TEST(SAVE, SaveGraph)
{
	std::string expect_pbfile = testdir + "/onnx.onnx";
	std::string got_pbfile = "got_onnx.onnx";

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


#endif // DISABLE_SAVE_TEST
