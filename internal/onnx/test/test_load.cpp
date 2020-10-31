
#ifndef DISABLE_ONNX_LOAD_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/teq/mock/mock.hpp"

#include "dbg/print/teq.hpp"

#include "internal/onnx/load.hpp"


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

	teq::TensptrT unmarsh_layr (std::string opname,
		const teq::TensptrT& root, const teq::TensptrT& child,
		marsh::Maps&& attrs) const override
	{
		// todo: implement mock layer
		return std::make_shared<MockFunctor>(teq::TensptrsT{child}, teq::Opcode{opname, 0});
	}
};


TEST(LOAD, BadGraph)
{
	{
		onnx::ModelProto model;
		std::fstream inputstr(testdir + "/bad_onnx.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
		MockUnmarshFuncs unmarsh;
		onnx::TensptrIdT ids;
		EXPECT_FATAL(onnx::load_graph(ids, model.graph(), unmarsh),
			"unknown onnx attribute type of `peanut`");
	}
	{
		onnx::ModelProto model;
		std::fstream inputstr(testdir + "/bad_onnx2.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
		MockUnmarshFuncs unmarsh;
		onnx::TensptrIdT ids;
		EXPECT_FATAL(onnx::load_graph(ids, model.graph(), unmarsh),
			"unknown graph attribute `peanut`");
	}
}


TEST(LOAD, SimpleGraph)
{
	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/simple_onnx.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	MockUnmarshFuncs unmarsh;
	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = onnx::load_graph(ids, model.graph(), unmarsh);
	EXPECT_EQ(2, graph_roots.size());

	std::string expect;
	std::string got;
	std::string line;
	std::ifstream expectstr(testdir + "/simple_onnx.txt");
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
	artist.cfg_.showshape_ = true;
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


TEST(LOAD, LayerGraph)
{
	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/layer_onnx.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	MockUnmarshFuncs unmarsh;
	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = onnx::load_graph(ids, model.graph(), unmarsh);
	EXPECT_EQ(2, graph_roots.size());

	std::string expect;
	std::string got;
	std::string line;
	std::ifstream expectstr(testdir + "/layer_onnx.txt");
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
	artist.cfg_.showshape_ = true;
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


TEST(LOAD, ReplaceLayerGraph)
{
	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/layer_onnx.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	MockUnmarshFuncs unmarsh;

	teq::TensptrT badm = std::make_shared<MockLeaf>(
		std::vector<double>{}, teq::Shape(), "bad_replaced");
	onnx::TensptrIdT badids;
	badids.insert({badm, "5"});
	EXPECT_FATAL(onnx::load_graph(badids, model.graph(), unmarsh),
		"duplicate id 5");

	teq::TensptrT m = std::make_shared<MockLeaf>(
		std::vector<double>{}, teq::Shape(), "replaced");
	onnx::TensptrIdT ids;
	ids.insert({m, "1"});
	teq::TensptrsT graph_roots = onnx::load_graph(ids, model.graph(), unmarsh);
	EXPECT_EQ(2, graph_roots.size());

	ASSERT_HAS(ids.right, "root1");
	ASSERT_HAS(ids.right, "root2");
}


TEST(LOAD, SimpleGraphEarlyStop)
{
	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/simple_stop.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	MockUnmarshFuncs unmarsh;
	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = onnx::load_graph(ids, model.graph(), unmarsh);
	EXPECT_EQ(2, graph_roots.size());

	std::string expect;
	std::string got;
	std::string line;
	std::ifstream expectstr(testdir + "/simple_stop.txt");
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
	artist.cfg_.showshape_ = true;
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


#endif // DISABLE_ONNX_LOAD_TEST
