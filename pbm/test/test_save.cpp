
#ifndef DISABLE_SAVE_TEST


#include <fstream>

#include <google/protobuf/util/message_differencer.h>

#include "gtest/gtest.h"

#include "teq/ifunctor.hpp"

#include "pbm/save.hpp"

#include "tag/prop.hpp"

#include "pbm/test/common.hpp"


const std::string testdir = "models/test";


struct TestSaver : public pbm::iSaver
{
	std::string save_leaf (teq::iLeaf* leaf) override
	{
		return std::string(leaf->shape().n_elems(), 0);
	}

	std::vector<double> save_shaper (const teq::CvrtptrT& mapper) override
	{
		std::vector<double> out;
		mapper->access(
			[&out](const teq::MatrixT& mat)
			{
				for (teq::RankT i = 0; i < teq::mat_dim; ++i)
				{
					for (teq::RankT j = 0; j < teq::mat_dim; ++j)
					{
						out.push_back(mat[i][j]);
					}
				}
			});
		return out;
	}

	std::vector<double> save_coorder (const teq::CvrtptrT& mapper) override
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
		std::vector<teq::TensptrT> roots;

		// subtree one
		teq::Shape shape({3, 7});
		teq::TensptrT osrc(new MockTensor(shape));

		teq::Shape shape2({7, 3});
		teq::TensptrT osrc2(new MockTensor(shape2));

		auto& preg = tag::get_property_reg();
		preg.property_tag(osrc, "osrc");
		preg.property_tag(osrc2, "osrc2");

		{
			teq::TensptrT src(new MockTensor(shape));

			teq::Shape shape3({3, 1, 7});
			teq::TensptrT src2(new MockTensor(shape3));

			teq::TensptrT dest(teq::Functor::get(teq::Opcode{"-", 0}, {
				{src2, teq::identity},
				{teq::TensptrT(teq::Functor::get(teq::Opcode{"@", 1}, {
					{teq::TensptrT(teq::Functor::get(teq::Opcode{"/", 2}, {
						{teq::TensptrT(teq::Functor::get(teq::Opcode{"neg", 3}, {
							{osrc, teq::identity},
						})), teq::identity},
						{teq::TensptrT(teq::Functor::get(teq::Opcode{"+", 4}, {
							{teq::TensptrT(
								teq::Functor::get(teq::Opcode{"sin", 5}, {
								{src, teq::identity}})), teq::identity},
							{src, teq::identity},
						})), teq::identity}
					})), teq::permute({1, 0})},
					{osrc2, teq::identity}
				})), teq::permute({1, 2, 0})},
			}));
			roots.push_back(dest);

			preg.property_tag(src, "subtree_src");
			preg.property_tag(src2, "subtree_src2");
			preg.property_tag(dest, "subtree_dest");
		}

		// subtree two
		{
			teq::Shape mshape({3, 3});
			teq::TensptrT src(new MockTensor(mshape));

			teq::TensptrT src2(new MockTensor(mshape));

			teq::TensptrT src3(new MockTensor(mshape));

			teq::TensptrT dest(teq::Functor::get(teq::Opcode{"-", 0}, {
				{src, teq::identity},
				{teq::TensptrT(teq::Functor::get(teq::Opcode{"*", 6}, {
					{teq::TensptrT(teq::Functor::get(teq::Opcode{"abs", 7}, {
						{src, teq::identity},
					})), teq::identity},
					{teq::TensptrT(teq::Functor::get(teq::Opcode{"exp", 8}, {
						{src2, teq::identity},
					})), teq::identity},
					{teq::TensptrT(teq::Functor::get(teq::Opcode{"neg", 3}, {
						{src3, teq::identity},
					})), teq::identity},
				})), teq::identity},
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

		saver.save(graph);

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
