
#ifndef DISABLE_LOAD_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "dbg/print/teq.hpp"

#include "onnx/load.hpp"


const std::string testdir = "models/test";

struct MockUnmarshFuncs final : public onnx::iUnmarshFuncs
{
	teq::TensptrT unmarsh_leaf (const onnx::TensorProto& tens,
		teq::Usage usage, std::string name) const override
	{
		return std::make_shared<MockLeaf>(
			std::vector<double>{}, onnx::unmarshal_shape(tens), name);
	}


	teq::TensptrT unmarsh_func (std::string opname,
		const teq::TensptrsT& edges, marsh::Maps&& attrs) const override
	{
		return std::make_shared<MockFunctor>(edges, teq::Opcode{opname, 0});
	}
};

TEST(LOAD, LoadGraph)
{
	onnx::GraphProto graph;
	{
		std::fstream inputstr(testdir + "/onnx.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(graph.ParseFromIstream(&inputstr));
	}

	MockUnmarshFuncs unmarsh;
	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = onnx::load_graph(ids, graph, unmarsh);
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


#endif // DISABLE_LOAD_TEST
