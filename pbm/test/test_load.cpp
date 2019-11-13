
#ifndef DISABLE_LOAD_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "dbg/stream/teq.hpp"

#include "tag/prop.hpp"

#include "pbm/load.hpp"

#include "pbm/test/common.hpp"


const std::string testdir = "models/test";


struct TestLoader : public pbm::iLoader
{
	teq::TensptrT generate_leaf (const char* pb, teq::Shape shape,
		std::string typelabel, std::string label, bool is_const) override
	{
		return teq::TensptrT(new MockTensor(shape));
	}

	teq::TensptrT generate_func (std::string opname, teq::ArgsT args) override
	{
		return teq::TensptrT(teq::Functor::get(teq::Opcode{opname, 0}, args));
	}

	teq::ShaperT generate_shaper (std::vector<double> coord) override
	{
		if (teq::mat_dim * teq::mat_dim != coord.size())
		{
			logs::fatal("cannot deserialize non-matrix coordinate map");
		}
		return std::make_shared<teq::ShapeMap>(
			[&](teq::MatrixT& fwd)
			{
				for (teq::RankT i = 0; i < teq::mat_dim; ++i)
				{
					for (teq::RankT j = 0; j < teq::mat_dim; ++j)
					{
						fwd[i][j] = coord[i * teq::mat_dim + j];
					}
				}
			});
	}

	teq::CoordptrT generate_coorder (
		std::string opname, std::vector<double> coord) override
	{
		return generate_shaper(coord);
	}
};


TEST(LOAD, LoadGraph)
{
	cortenn::Graph graph;
	{
		std::fstream inputstr(testdir + "/pbm_test.pbx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(graph.ParseFromIstream(&inputstr));
	}

	teq::TensptrSetT graphinfo;
	pbm::load_graph<TestLoader>(graphinfo, graph);
	EXPECT_EQ(2, graphinfo.size());

	auto& reg = tag::get_reg();
	tag::Query q;

	std::vector<std::string> root_props;
	std::unordered_map<std::string,teq::TensptrT> propdtens;
	for (auto tens : graphinfo)
	{
		tens->accept(q);
		auto tags = reg.get_tags(tens.get());
		ASSERT_HAS(tags, tag::props_key);
		auto& props = tags[tag::props_key];
		ASSERT_EQ(1, props.size());
		propdtens.emplace(props[0], tens);
		root_props.insert(root_props.end(), props.begin(), props.end());
	}
	EXPECT_ARRHAS(root_props, "subtree_dest");
	EXPECT_ARRHAS(root_props, "subtree2_dest");

	ASSERT_HAS(q.labels_, tag::props_key);
	auto& props = q.labels_[tag::props_key];

	ASSERT_HAS(props, "subtree_src");
	ASSERT_HAS(props, "subtree_src2");
	ASSERT_HAS(props, "subtree2_src");
	ASSERT_HAS(props, "subtree2_src2");
	ASSERT_HAS(props, "subtree2_src3");
	ASSERT_HAS(props, "osrc");
	ASSERT_HAS(props, "osrc2");

	auto& sts = props["subtree_src"];
	auto& sts2 = props["subtree_src2"];
	auto& st2s = props["subtree2_src"];
	auto& st2s2 = props["subtree2_src2"];
	auto& st2s3 = props["subtree2_src3"];
	auto& os = props["osrc"];
	auto& os2 = props["osrc2"];

	ASSERT_EQ(1, sts.size());
	ASSERT_EQ(1, sts2.size());
	ASSERT_EQ(1, st2s.size());
	ASSERT_EQ(1, st2s2.size());
	ASSERT_EQ(1, st2s3.size());
	ASSERT_EQ(1, os.size());
	ASSERT_EQ(1, os2.size());

	ASSERT_HAS(propdtens, "subtree_dest");
	ASSERT_HAS(propdtens, "subtree2_dest");
	teq::TensptrT tree1 = propdtens["subtree_dest"];
	teq::TensptrT tree2 = propdtens["subtree2_dest"];

	ASSERT_NE(nullptr, tree1);
	ASSERT_NE(nullptr, tree2);

	std::string expect;
	std::string got;
	std::string line;
	std::ifstream expectstr(testdir + "/pbm_test.txt");
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
