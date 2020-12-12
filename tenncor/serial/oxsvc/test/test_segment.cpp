
#ifndef DISABLE_OXSVC_SEGMENT_TEST


#include <fstream>

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/global/mock/mock.hpp"

#include "dbg/print/teq.hpp"
#include "dbg/print/printsvc/mock/mock.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/mock/mock.hpp"
#include "tenncor/distr/iosvc/mock/mock.hpp"
#include "tenncor/serial/oxsvc/mock/mock.hpp"


#ifdef CMAKE_SOURCE_DIR
const std::string testdir = std::string(CMAKE_SOURCE_DIR) + "models/test";
#else
const std::string testdir = "models/test";
#endif


const std::string test_service = "tenncor.serial.oxsvc.test";


struct SEGMENT : public ::testing::Test, public DistrTestcase
{
protected:
	distr::iDistrMgrptrT make_mgr (const std::string& id)
	{
		return make_mgr(id, reserve_port());
	}

	distr::iDistrMgrptrT make_mgr (const std::string& id, size_t port)
	{
		return DistrTestcase::make_local_mgr(port, {
			register_mock_iosvc,
			register_mock_oxsvc,
			register_mock_printsvc,
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
	// _`--(12)
	// _`--(14)
	// _____`--(1)
	// _____|___`--(8)
	// _____|___|___`--(7)
	// _____|___`--(11)
	// _____|_______`--(10)
	// _____|_______|___`--(9)
	// _____|_______`--(9)
	// _____`--(13)

	// (root2)
	// _`--(3)
	// _`--(18)
	// _____`--(2)
	// _____|___`--(3)
	// _____`--(16)
	// _____|___`--(15)
	// _____`--(17)
	// _________`--(4)

	size_t root1v = vertices.at("root1");
	size_t root2v = vertices.at("root2");
	size_t v12 = vertices.at("12");
	size_t v14 = vertices.at("14");
	size_t v11 = vertices.at("11");
	size_t v10 = vertices.at("10");
	size_t v9 = vertices.at("9");

	EXPECT_LE(18, dist(root1v, root2v)); // # of nodes denotes infinity
	EXPECT_EQ(2, dist(v12, v14));
	EXPECT_EQ(4, dist(v12, v11));
	EXPECT_EQ(2, dist(v11, v14));
	EXPECT_EQ(1, dist(v9, v11));
	EXPECT_EQ(1, dist(v9, v10));
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
	// _`--(12)
	// _`--(14)
	// _____`--(1)
	// _____|___`--(8)
	// _____|___|___`--(7)
	// _____|___`--(11)
	// _____|_______`--(10)
	// _____|_______|___`--(9)
	// _____|_______`--(9)
	// _____`--(13)
	ASSERT_EQ(10, graph1.size());
	EXPECT_HAS(graph1, "12");
	EXPECT_HAS(graph1, "14");
	EXPECT_HAS(graph1, "1");
	EXPECT_HAS(graph1, "8");
	EXPECT_HAS(graph1, "7");
	EXPECT_HAS(graph1, "11");
	EXPECT_HAS(graph1, "10");
	EXPECT_HAS(graph1, "9");
	EXPECT_HAS(graph1, "13");

	// (root2)
	// _`--(3)
	// _`--(18)
	// _____`--(2)
	// _____|___`--(3)
	// _____`--(16)
	// _____|___`--(15)
	// _____`--(17)
	// _________`--(4)
	ASSERT_EQ(8, graph2.size());
	EXPECT_HAS(graph2, "3");
	EXPECT_HAS(graph2, "18");
	EXPECT_HAS(graph2, "2");
	EXPECT_HAS(graph2, "16");
	EXPECT_HAS(graph2, "15");
	EXPECT_HAS(graph2, "17");
	EXPECT_HAS(graph2, "4");

}


TEST_F(SEGMENT, TwoMeans)
{
	distr::iDistrMgrptrT manager(make_mgr("mgr"));
	distr::iDistrMgrptrT manager2(make_mgr("mgr2"));
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
	// _`--(12)
	// _`--(14)
	// _____`--(1)
	// _____|___`--(8)
	// _____|___|___`--(7)
	// _____|___`--(11)
	// _____|_______`--(10)
	// _____|_______|___`--(9)
	// _____|_______`--(9)
	// _____`--(13)
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
				vertices.at("8"),
			};
		});
	ASSERT_EQ(2, kselected);

	ASSERT_EQ(2, topography.size());
	ASSERT_HAS(topography, "root1");
	ASSERT_HAS(topography, "8");
	EXPECT_STREQ("mgr", topography.at("root1").c_str());
	EXPECT_STREQ("mgr2", topography.at("8").c_str());

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
