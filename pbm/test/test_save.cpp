
#ifndef DISABLE_SAVE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "pbm/save.hpp"

#include "tag/prop.hpp"


const std::string testdir = "models/test";


static std::string save_leaf (teq::iLeaf* leaf)
{
	return std::string(leaf->shape().n_elems(), 0);
}


TEST(SAVE, SaveGraph)
{
	std::string expect_pbfile = testdir + "/pbm_test.pbx";
	// std::string got_pbfile = "got_pbm_test.pbx";
	std::string got_pbfile = "/tmp/pbm_test.pbx";

	{
		tenncor::Graph graph;
		std::vector<teq::TensptrT> roots;

		// subtree one
		teq::Shape shape({3, 7});
		teq::Shape shape2({7, 3});
		teq::TensptrT osrc = std::make_shared<MockTensor>(shape, "osrc");
		teq::TensptrT osrc2 = std::make_shared<MockTensor>(shape2, "osrc2");

		auto& preg = tag::get_property_reg();
		preg.property_tag(osrc, "osrc");
		preg.property_tag(osrc2, "osrc2");

		{
			teq::Shape shape3({3, 1, 7});
			teq::TensptrT src = std::make_shared<MockTensor>(shape, "src");
			teq::TensptrT src2 = std::make_shared<MockTensor>(shape3, "src2");

			teq::TensptrT dest = std::make_shared<MockFunctor>(MockEdgesT{
				MockEdge(src2, {}),
				MockEdge(std::make_shared<MockFunctor>(MockEdgesT{
					MockEdge(std::make_shared<MockFunctor>(MockEdgesT{
						MockEdge(std::make_shared<MockFunctor>(MockEdgesT{
							MockEdge(osrc, {}),
						}, teq::Opcode{"neg", 3}), {}),
						MockEdge(std::make_shared<MockFunctor>(MockEdgesT{
							MockEdge(std::make_shared<MockFunctor>(MockEdgesT{
								MockEdge(src, {}),
							}, teq::Opcode{"sin", 5}), {}),
							{src, {}},
						}, teq::Opcode{"+", 4}), {})
					}, teq::Opcode{"/", 2}), {}, {1, 0}),
					MockEdge(osrc2, {})
				}, teq::Opcode{"@", 1}), {}, {1, 2, 0}),
			}, teq::Opcode{"-", 0});
			roots.push_back(dest);

			preg.property_tag(src, "subtree_src");
			preg.property_tag(src2, "subtree_src2");
			preg.property_tag(dest, "subtree_dest");
		}

		// subtree two
		{
			teq::Shape mshape({3, 3});
			teq::TensptrT src = std::make_shared<MockTensor>(mshape, "s2src");
			teq::TensptrT src2 = std::make_shared<MockTensor>(mshape, "s2src2");
			teq::TensptrT src3 = std::make_shared<MockTensor>(mshape, "s2src3");

			teq::TensptrT dest = std::make_shared<MockFunctor>(MockEdgesT{
				MockEdge(src, {}),
				MockEdge(std::make_shared<MockFunctor>(MockEdgesT{
					MockEdge(std::make_shared<MockFunctor>(MockEdgesT{
						MockEdge(src, {}),
					}, teq::Opcode{"abs", 7}), {}),
					MockEdge(std::make_shared<MockFunctor>(MockEdgesT{
						MockEdge(src2, {}),
					}, teq::Opcode{"exp", 8}), {}),
					MockEdge(std::make_shared<MockFunctor>(MockEdgesT{
						MockEdge(src3, {}),
					}, teq::Opcode{"neg", 3}), {}),
				}, teq::Opcode{"*", 6}), {}),
			}, teq::Opcode{"-", 0});
			roots.push_back(dest);

			preg.property_tag(src, "subtree2_src");
			preg.property_tag(src2, "subtree2_src2");
			preg.property_tag(src3, "subtree2_src3");
			preg.property_tag(dest, "subtree2_dest");
		}

		pbm::save_graph(graph, roots, preg.tag_reg_, save_leaf);

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

		tenncor::Graph expect_graph;
		tenncor::Graph got_graph;
		ASSERT_TRUE(expect_graph.ParseFromIstream(&expect_ifs));
		ASSERT_TRUE(got_graph.ParseFromIstream(&got_ifs));

		google::protobuf::util::MessageDifferencer differ;
		std::string report;
		differ.ReportDifferencesToString(&report);
		EXPECT_TRUE(differ.Compare(expect_graph, got_graph)) << report;
	}
}


#endif // DISABLE_SAVE_TEST
