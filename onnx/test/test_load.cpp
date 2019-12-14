
#ifndef DISABLE_LOAD_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "dbg/stream/teq.hpp"

#include "onnx/load.hpp"


const std::string testdir = "models/test";


static teq::TensptrT generate_leaf (const onnx::TensorProto& tens,
	bool is_const, std::string name)
{
	return std::make_shared<MockLeaf>(onnx::unmarshal_shape(tens), name);
}


static teq::TensptrT generate_func (std::string opname,
	const teq::TensptrsT& edges, marsh::Maps&& attrs)
{
	return std::make_shared<MockFunctor>(edges, teq::Opcode{opname, 0});
}


TEST(LOAD, LoadGraph)
{
	onnx::GraphProto graph;
	{
		std::fstream inputstr(testdir + "/onnx.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(graph.ParseFromIstream(&inputstr));
	}

	teq::TensptrsT graph_roots;
	onnx::load_graph(graph_roots, graph,
		generate_leaf, generate_func);
	EXPECT_EQ(2, graph_roots.size());

	std::string expect;
	std::string got;
	std::string line;
	std::ifstream expectstr(testdir + "/onnx.txt");
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
	artist.showshape_ = true;
	std::stringstream gotstr;
	for (auto tens : graph_roots)
	{
		ASSERT_NE(nullptr, tens);
		artist.print(gotstr, tens);
	}

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
