
#ifndef DISABLE_OXSVC_SEGMENT_TEST


#include <fstream>

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/global/mock/mock.hpp"

#include "dbg/print/teq.hpp"
#include "dbg/print/printsvc/printsvc.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/serial/oxsvc/oxsvc.hpp"


const std::string testdir = "models/test";


const std::string test_service = "tenncor.serial.oxsvc.test";


struct SEGMENT : public ::testing::Test, public DistrTestcase
{
protected:
	distr::iDistrMgrptrT make_mgr (size_t port, const std::string& id = "")
	{
		return DistrTestcase::make_mgr(port, {
			distr::register_iosvc,
			distr::register_oxsvc,
			distr::register_printsvc,
		}, id);
	}
};


TEST_F(SEGMENT, MinDist)
{
	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/remote_oxsvc.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	distr::ox::GraphT nodes;
	distr::ox::extract_nodes(nodes, model.graph());

	types::StrUMapT<size_t> vertices;
	std::vector<size_t> distance;
	segment::floyd_warshall(distance, vertices, nodes);

	auto dist = segment::sqr_curry(distance, nodes.size());

	// (root1)
	// _`--(10)
	// _`--(12)
	// _____`--(1)
	// _____|___`--(6)
	// _____|___|___`--(5)
	// _____|___`--(9)
	// _____|_______`--(8)
	// _____|_______|___`--(7)
	// _____|_______`--(7)
	// _____`--(11)

	// (root2)
	// _`--(3)
	// _`--(16)
	// _____`--(2)
	// _____|___`--(3)
	// _____`--(14)
	// _____|___`--(13)
	// _____`--(15)
	// _________`--(4)

	size_t root1v = vertices.at("root1");
	size_t root2v = vertices.at("root2");
	size_t v10 = vertices.at("10");
	size_t v12 = vertices.at("12");
	size_t v9 = vertices.at("9");
	size_t v8 = vertices.at("8");
	size_t v7 = vertices.at("7");

	EXPECT_LE(18, dist(root1v, root2v)); // # of nodes denotes infinity
	EXPECT_EQ(2, dist(v10, v12));
	EXPECT_EQ(4, dist(v10, v9));
	EXPECT_EQ(2, dist(v9, v12));
	EXPECT_EQ(1, dist(v7, v9));
	EXPECT_EQ(1, dist(v7, v8));
}


TEST_F(SEGMENT, Disjoint)
{
	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/remote_oxsvc.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	distr::ox::GraphT nodes;
	distr::ox::extract_nodes(nodes, model.graph());

	auto graphs = segment::disjoint_graphs(nodes);
	ASSERT_EQ(2, graphs.size());

	distr::ox::GraphT graph1, graph2;
	for (auto& graph : graphs)
	{
		if (estd::has(graph, "root1"))
		{
			graph1 = graph;
		}
		else if (estd::has(graph, "root2"))
		{
			graph2 = graph;
		}
		else
		{
			FAIL() << "failed to find find graph containing root1 or root2";
		}
	}

	// (root1)
	// _`--(10)
	// _`--(12)
	// _____`--(1)
	// _____|___`--(6)
	// _____|___|___`--(5)
	// _____|___`--(9)
	// _____|_______`--(8)
	// _____|_______|___`--(7)
	// _____|_______`--(7)
	// _____`--(11)
	ASSERT_EQ(10, graph1.size());
	EXPECT_HAS(graph1, "10");
	EXPECT_HAS(graph1, "12");
	EXPECT_HAS(graph1, "1");
	EXPECT_HAS(graph1, "6");
	EXPECT_HAS(graph1, "5");
	EXPECT_HAS(graph1, "9");
	EXPECT_HAS(graph1, "8");
	EXPECT_HAS(graph1, "7");
	EXPECT_HAS(graph1, "1");

	// (root2)
	// _`--(3)
	// _`--(16)
	// _____`--(2)
	// _____|___`--(3)
	// _____`--(14)
	// _____|___`--(13)
	// _____`--(15)
	// _________`--(4)
	ASSERT_EQ(8, graph2.size());
	EXPECT_HAS(graph2, "3");
	EXPECT_HAS(graph2, "16");
	EXPECT_HAS(graph2, "2");
	EXPECT_HAS(graph2, "14");
	EXPECT_HAS(graph2, "13");
	EXPECT_HAS(graph2, "15");
	EXPECT_HAS(graph2, "4");

}


TEST_F(SEGMENT, TwoMeans)
{
	distr::iDistrMgrptrT manager(make_mgr(5112, "mgr"));
	distr::iDistrMgrptrT manager2(make_mgr(5113, "mgr2"));
	global::set_generator(std::make_shared<MockGenerator>());

	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/remote_oxsvc.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	distr::ox::GraphT nodes;
	distr::ox::extract_nodes(nodes, model.graph());

	auto graphs = segment::disjoint_graphs(nodes);
	ASSERT_EQ(2, graphs.size());

	// (root1)
	// _`--(10)
	// _`--(12)
	// _____`--(1)
	// _____|___`--(6)
	// _____|___|___`--(5)
	// _____|___`--(9)
	// _____|_______`--(8)
	// _____|_______|___`--(7)
	// _____|_______`--(7)
	// _____`--(11)
	distr::ox::GraphT graph1, graph2;
	for (auto& graph : graphs)
	{
		if (estd::has(graph, "root1"))
		{
			graph1 = graph;
		}
		else if (estd::has(graph, "root2"))
		{
			graph2 = graph;
		}
		else
		{
			FAIL() << "failed to find find graph containing root1 or root2";
		}
	}

	// if both selected nodes are in the same disjoint graph
	size_t kselected = 0;
	distr::ox::TopographyT topography = segment::kmeans(
		{"mgr", "mgr2"}, graph1,
		[&kselected](size_t k, const types::StrUMapT<size_t>& vertices)
		{
			kselected = k;
			return std::vector<size_t>{
				vertices.at("1"),
				vertices.at("6"),
			};
		});
	ASSERT_EQ(2, kselected);

	ASSERT_EQ(2, topography.size());
	EXPECT_HAS(topography, "root1");
	EXPECT_HAS(topography, "6");
	EXPECT_STREQ("mgr", topography.at("root1").c_str());
	EXPECT_STREQ("mgr2", topography.at("6").c_str());

	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = distr::get_oxsvc(*manager).load_graph(
		ids, model.graph(), topography);
	ASSERT_EQ(2, graph_roots.size());

	teq::TensptrT root1, root2;
	for (auto root : graph_roots)
	{
		auto id = distr::get_iosvc(*manager).lookup_id(root.get());
		ASSERT_TRUE(id);
		if ("root1" == *id)
		{
			root1 = root;
		}
		else if ("root2" == *id)
		{
			root2 = root;
		}
		else
		{
			FAIL() << "failed to find find graph containing root1 or root2";
		}
	}

	EXPECT_EQ(nullptr, dynamic_cast<distr::iDistrRef*>(root1.get()));

	std::stringstream ss;
	distr::get_printsvc(*manager).print_ascii(ss, root1.get());
	std::string expect =
		"(SUB)\n"
		"_`--(variable:src2)\n"
		"_`--(POW)\n"
		"_____`--(DIV)\n"
		"_____|___`--[mgr2]:(NEG)\n"
		"_____|___|___`--(variable:osrc)\n"
		"_____|___`--(ADD)\n"
		"_____|_______`--(SIN)\n"
		"_____|_______|___`--(variable:src)\n"
		"_____|_______`--(variable:src)\n"
		"_____`--(variable:osrc2)\n";
	EXPECT_STREQ(expect.c_str(), ss.str().c_str());
}


#endif // DISABLE_OXSVC_SEGMENT_TEST
