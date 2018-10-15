#include <fstream>

#include "gtest/gtest.h"

#include "llo/api.hpp"

#include "pbm/graph.hpp"


#ifndef DISABLE_SAVE_TEST


const std::string testdir = "pbm/test/data";


TEST(SAVE, SaveGraph)
{
	std::string expect_pbfile = testdir + "/graph.pb";
	std::string got_pbfile = "got_graph.pb";
	tenncor::Graph graph;
	std::vector<llo::DataNode> roots;

	// subtree one
	ade::Shape shape({3, 7});
	std::vector<double> odata = {69, 49, 43, 96, 38, 21, 6, 26, 26, 57, 46, 69,
		98, 66, 98, 84, 5, 78, 82, 95, 98};
	auto osrc = llo::Source<double>::get(shape, odata);

	{
		std::vector<double> data = {16, 65, 57, 11, 10, 17, 76, 47, 47, 44, 47, 14,
			9, 54, 35, 94, 15, 93, 43, 56, 50};
		auto src = llo::Source<double>::get(shape, data);
		auto dest = llo::div(llo::neg(osrc), llo::add(llo::sin(src), src));
		roots.push_back(dest);
	}

	// subtree two
	{
		ade::Shape mshape({11, 3});
		std::vector<double> data = {90, 34, 15, 21, 69, 24, 34, 16, 18, 51, 59,
			80, 34, 60, 82, 42, 4, 68, 99, 90, 98, 1, 98, 81, 43, 48, 26, 17, 75,
			69, 7, 66, 23};
		auto src = llo::Source<double>::get(mshape, data);

		ade::Shape mshape2({7, 2});
		std::vector<double> data2 = {71, 55, 48, 72, 43, 8, 71, 86, 43, 44, 25,
			50, 62, 66};
		auto src2 = llo::Source<double>::get(mshape2, data2);

		auto dest = llo::matmul(src2, llo::matmul(osrc, src));
		roots.push_back(dest);
	}

	save_graph(graph, roots);
	{
		std::fstream gotstr(got_pbfile,
			std::ios::out | std::ios::trunc | std::ios::binary);
		ASSERT_TRUE(gotstr.is_open());
		ASSERT_TRUE(graph.SerializeToOstream(&gotstr));
	}

	std::fstream expect_ifs(expect_pbfile, std::ios::in | std::ios::binary);
	std::fstream got_ifs(got_pbfile, std::ios::in | std::ios::binary);
	ASSERT_TRUE(expect_ifs.is_open());
	ASSERT_TRUE(got_ifs.is_open());

	std::string expect;
	std::string got;
	// skip the first line (it contains timestamp)
	expect_ifs >> expect;
	got_ifs >> got;
	for (size_t lineno = 1; expect_ifs && got_ifs; ++lineno)
	{
		expect_ifs >> expect;
		got_ifs >> got;
		EXPECT_STREQ(expect.c_str(), got.c_str()) << "line number " << lineno;
	}
}


#endif // DISABLE_SAVE_TEST
