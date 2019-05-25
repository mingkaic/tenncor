
#ifndef DISABLE_LOAD_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "dbg/stream/ade.hpp"

#include "pbm/load.hpp"

#include "pbm/test/common.hpp"


const std::string testdir = "pbm/data";


struct TestLoader : public pbm::iLoader
{
	ade::TensptrT generate_leaf (const char* pb, ade::Shape shape,
		std::string typelabel, std::string label, bool is_const) override
	{
		return ade::TensptrT(new MockTensor(shape));
	}

	ade::TensptrT generate_func (std::string opname, ade::ArgsT args) override
	{
		return ade::TensptrT(ade::Functor::get(ade::Opcode{opname, 0}, args));
	}

	ade::CoordptrT generate_shaper (std::vector<double> coord) override
	{
		if (ade::mat_dim * ade::mat_dim != coord.size())
		{
			logs::fatal("cannot deserialize non-matrix coordinate map");
		}
		return std::make_shared<ade::CoordMap>(
			[&](ade::MatrixT fwd)
			{
				for (uint8_t i = 0; i < ade::mat_dim; ++i)
				{
					for (uint8_t j = 0; j < ade::mat_dim; ++j)
					{
						fwd[i][j] = coord[i * ade::mat_dim + j];
					}
				}
			});
	}

	ade::CoordptrT generate_coorder (
		std::string opname, std::vector<double> coord) override
	{
		return generate_shaper(coord);
	}
};


TEST(LOAD, LoadGraph)
{
	cortenn::Graph graph;
	{
		std::fstream inputstr(testdir + "/graph.pb",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(graph.ParseFromIstream(&inputstr));
	}

	pbm::GraphInfo graphinfo;
	pbm::load_graph<TestLoader>(graphinfo, graph);

	EXPECT_EQ(2, graphinfo.roots_.size());

	ASSERT_EQ(3, graphinfo.tens_.children_.size());
	ASSERT_EQ(0, graphinfo.tens_.tens_.size());

	auto global_it = graphinfo.tens_.children_.find("global");
	auto subtree_it = graphinfo.tens_.children_.find("subtree");
	auto subtree2_it = graphinfo.tens_.children_.find("subtree2");

	ASSERT_NE(graphinfo.tens_.children_.end(), global_it) << "global namespace not found";
	ASSERT_NE(graphinfo.tens_.children_.end(), subtree_it) << "subtree namespace not found";
	ASSERT_NE(graphinfo.tens_.children_.end(), subtree2_it) << "subtree2 namespace not found";

	auto subtree = subtree_it->second;
	auto subtree2 = subtree2_it->second;
	ASSERT_EQ(3, subtree->tens_.size());
	ASSERT_EQ(4, subtree2->tens_.size());

	auto dest_it = subtree->tens_.find("dest");
	auto dest2_it = subtree2->tens_.find("dest");
	ASSERT_NE(subtree->tens_.end(), dest_it) << "{subtree, dest} not found";
	ASSERT_NE(subtree2->tens_.end(), dest2_it) << "{subtree2, dest} not found";

	ade::TensptrT tree1 = graphinfo.tens_.get_labelled({"subtree", "dest"});
	ade::TensptrT tree2 = graphinfo.tens_.get_labelled({"subtree2", "dest"});

	ASSERT_NE(nullptr, tree1);
	ASSERT_NE(nullptr, tree2);

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
	artist.print(gotstr, tree1);
	artist.print(gotstr, tree2);

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


#endif // DISABLE_LOAD_TEST
