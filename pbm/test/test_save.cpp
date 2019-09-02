
#ifndef DISABLE_SAVE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "ade/functor.hpp"

#include "pbm/save.hpp"

#include "tag/prop.hpp"

#include "pbm/test/common.hpp"


const std::string testdir = "models/test";


struct TestSaver : public pbm::iSaver
{
	std::string save_leaf (ade::iLeaf* leaf) override
	{
		return std::string(leaf->shape().n_elems(), 0);
	}

	std::vector<double> save_shaper (const ade::CoordptrT& mapper) override
	{
		std::vector<double> out;
		mapper->access(
			[&out](const ade::MatrixT& mat)
			{
				for (ade::RankT i = 0; i < ade::mat_dim; ++i)
				{
					for (ade::RankT j = 0; j < ade::mat_dim; ++j)
					{
						out.push_back(mat[i][j]);
					}
				}
			});
		return out;
	}

	std::vector<double> save_coorder (const ade::CoordptrT& mapper) override
	{
		return save_shaper(mapper);
	}
};


TEST(SAVE, SaveGraph)
{
	std::string expect_pbfile = testdir + "/pbm_test.pbx";
	std::string got_pbfile = "got_pbm_test.pbx";

	{
		cortenn::Graph graph;
		std::vector<ade::TensptrT> roots;

		// subtree one
		ade::Shape shape({3, 7});
		ade::TensptrT osrc(new MockTensor(shape));

		ade::Shape shape2({7, 3});
		ade::TensptrT osrc2(new MockTensor(shape2));

		auto& preg = tag::get_property_reg();
		preg.property_tag(osrc, "osrc");
		preg.property_tag(osrc2, "osrc2");

		{
			ade::TensptrT src(new MockTensor(shape));

			ade::Shape shape3({3, 1, 7});
			ade::TensptrT src2(new MockTensor(shape3));

			ade::TensptrT dest(ade::Functor::get(ade::Opcode{"-", 0}, {
				{src2, ade::identity},
				{ade::TensptrT(ade::Functor::get(ade::Opcode{"@", 1}, {
					{ade::TensptrT(ade::Functor::get(ade::Opcode{"/", 2}, {
						{ade::TensptrT(ade::Functor::get(ade::Opcode{"neg", 3}, {
							{osrc, ade::identity},
						})), ade::identity},
						{ade::TensptrT(ade::Functor::get(ade::Opcode{"+", 4}, {
							{ade::TensptrT(
								ade::Functor::get(ade::Opcode{"sin", 5}, {
								{src, ade::identity}})), ade::identity},
							{src, ade::identity},
						})), ade::identity}
					})), ade::permute({1, 0})},
					{osrc2, ade::identity}
				})), ade::permute({1, 2, 0})},
			}));
			roots.push_back(dest);

			preg.property_tag(src, "subtree_src");
			preg.property_tag(src2, "subtree_src2");
			preg.property_tag(dest, "subtree_dest");
		}

		// subtree two
		{
			ade::Shape mshape({3, 3});
			ade::TensptrT src(new MockTensor(mshape));

			ade::TensptrT src2(new MockTensor(mshape));

			ade::TensptrT src3(new MockTensor(mshape));

			ade::TensptrT dest(ade::Functor::get(ade::Opcode{"-", 0}, {
				{src, ade::identity},
				{ade::TensptrT(ade::Functor::get(ade::Opcode{"*", 6}, {
					{ade::TensptrT(ade::Functor::get(ade::Opcode{"abs", 7}, {
						{src, ade::identity},
					})), ade::identity},
					{ade::TensptrT(ade::Functor::get(ade::Opcode{"exp", 8}, {
						{src2, ade::identity},
					})), ade::identity},
					{ade::TensptrT(ade::Functor::get(ade::Opcode{"neg", 3}, {
						{src3, ade::identity},
					})), ade::identity},
				})), ade::identity},
			}));
			roots.push_back(dest);

			preg.property_tag(src, "subtree2_src");
			preg.property_tag(src2, "subtree2_src2");
			preg.property_tag(src3, "subtree2_src3");
			preg.property_tag(dest, "subtree2_dest");
		}

		pbm::GraphSaver<TestSaver> saver;
		for (auto& root : roots)
		{
			root->accept(saver);
		}

		pbm::PathedMapT labels;
		saver.save(graph, labels);

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


#endif // DISABLE_SAVE_TEST
