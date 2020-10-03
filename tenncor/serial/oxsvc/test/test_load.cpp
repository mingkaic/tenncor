
#ifndef DISABLE_OXSVC_LOAD_TEST


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


struct LOAD : public ::testing::Test, public DistrTestcase
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


TEST_F(LOAD, AllLocalGraph)
{
	distr::iDistrMgrptrT manager(make_mgr(5112, "mgr"));

	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/local_oxsvc.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = distr::get_oxsvc(*manager).load_graph(ids, model.graph());
	EXPECT_EQ(2, graph_roots.size());

	PrettyEquation artist;
	artist.cfg_.showshape_ = true;
	std::stringstream gotstr;

	ASSERT_HAS(ids.right, "root1");
	ASSERT_HAS(ids.right, "root2");
	auto root1 = ids.right.at("root1");
	auto root2 = ids.right.at("root2");
	ASSERT_NE(nullptr, root1);
	ASSERT_NE(nullptr, root2);

	EXPECT_GRAPHEQ(
		"(SUB<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(variable:src2<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(POW<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(DIV<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(NEG<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___`--(variable:osrc<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(ADD<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(SIN<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|___`--(variable:src<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(variable:src<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:osrc2<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n", root1);
	EXPECT_GRAPHEQ(
		"(SUB<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(variable:s2src<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(ABS<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:s2src<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXP<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:s2src2<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(NEG<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(variable:s2src3<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n", root2);
}


TEST_F(LOAD, SimpleRemoteGraph)
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

	distr::ox::TopographyT topography = {
		{"root1", "mgr"},
		{"root2", "mgr"},
		{"1", "mgr2"},
		{"2", "mgr2"},
		{"3", "mgr2"},
		{"4", "mgr2"},
	};
	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = distr::get_oxsvc(*manager).load_graph(
		ids, model.graph(), topography);
	EXPECT_EQ(2, graph_roots.size());

	ASSERT_HAS(ids.right, "root1");
	ASSERT_HAS(ids.right, "root2");
	auto root1 = ids.right.at("root1");
	auto root2 = ids.right.at("root2");
	ASSERT_NE(nullptr, root1);
	ASSERT_NE(nullptr, root2);

	std::stringstream ss;
	distr::get_printsvc(*manager).print_ascii(ss, root1.get());
	std::string expect =
		"(SUB)\n"
		"_`--(variable:src2)\n"
		"_`--(POW)\n"
		"_____`--[mgr2]:(DIV)\n"
		"_____|___`--(NEG)\n"
		"_____|___|___`--(variable:osrc)\n"
		"_____|___`--(ADD)\n"
		"_____|_______`--(SIN)\n"
		"_____|_______|___`--(variable:src)\n"
		"_____|_______`--(variable:src)\n"
		"_____`--(variable:osrc2)\n";
	EXPECT_STREQ(expect.c_str(), ss.str().c_str());

	std::stringstream ss2;
	distr::get_printsvc(*manager).print_ascii(ss2, root2.get());
	std::string expect2 =
		"(SUB)\n"
		"_`--[mgr2]:(variable:s2src)\n"
		"_`--(MUL)\n"
		"_____`--[mgr2]:(ABS)\n"
		"_____|___`--(variable:s2src)\n"
		"_____`--(EXP)\n"
		"_____|___`--(variable:s2src2)\n"
		"_____`--(NEG)\n"
		"_________`--[mgr2]:(variable:s2src3)\n";
	EXPECT_STREQ(expect2.c_str(), ss2.str().c_str());
}


TEST_F(LOAD, FlattenRemoteGraph)
{
	distr::iDistrMgrptrT manager(make_mgr(5112, "mgr"));
	distr::iDistrMgrptrT manager2(make_mgr(5113, "mgr2"));
	global::set_generator(std::make_shared<MockGenerator>());
	auto& svc2 = distr::get_iosvc(*manager2);

	onnx::ModelProto model;
	{
		std::fstream inputstr(testdir + "/remote_oxsvc.onnx",
			std::ios::in | std::ios::binary);
		ASSERT_TRUE(inputstr.is_open());
		ASSERT_TRUE(model.ParseFromIstream(&inputstr));
	}

	// load nodes into a flat remote peer
	distr::ox::TopographyT topography = {
		{"root1", "mgr2"},
		{"root2", "mgr2"},
	};
	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = distr::get_oxsvc(*manager).load_graph(
		ids, model.graph(), topography);
	EXPECT_EQ(2, graph_roots.size());

	distr::iDistrRef* root1_ref;
	distr::iDistrRef* root2_ref;
	for (auto root : graph_roots)
	{
		auto ref = dynamic_cast<distr::iDistrRef*>(root.get());
		ASSERT_NE(nullptr, ref);
		auto id = ref->node_id();
		if (id == "root1")
		{
			root1_ref = ref;
		}
		else if (id == "root2")
		{
			root2_ref = ref;
		}
		else
		{
			FAIL() << "unexpected reference " << id;
		}
	}

	error::ErrptrT err = nullptr;
	auto root1 = svc2.lookup_node(err, root1_ref->node_id());
	ASSERT_NOERR(err);
	auto root2 = svc2.lookup_node(err, root2_ref->node_id());
	ASSERT_NOERR(err);

	EXPECT_GRAPHEQ(
		"(SUB<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(variable:src2<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(POW<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(DIV<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(NEG<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___`--(variable:osrc<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(ADD<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(SIN<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|___`--(variable:src<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(variable:src<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:osrc2<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n", root1);
	EXPECT_GRAPHEQ(
		"(SUB<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(variable:s2src<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(ABS<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:s2src<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXP<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:s2src2<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(NEG<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(variable:s2src3<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n", root2);
}


TEST_F(LOAD, CyclicRemoteGraph)
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

	// load nodes into a flat remote peer
	distr::ox::TopographyT topography = {
		{"root1", "mgr2"},
		{"12", "mgr"},
		{"1", "mgr2"},
		{"6", "mgr"},
		{"root2", "mgr"},
		{"16", "mgr2"},
		{"2", "mgr"},
	};
	onnx::TensptrIdT ids;
	teq::TensptrsT graph_roots = distr::get_oxsvc(*manager).load_graph(
		ids, model.graph(), topography);
	EXPECT_EQ(2, graph_roots.size());

	error::ErrptrT err = nullptr;
	auto root1ref = distr::get_iosvc(*manager).lookup_node(err, "root1");
	ASSERT_NOERR(err);
	auto root1 = distr::get_iosvc(*manager2).lookup_node(err, "root1");
	ASSERT_NOERR(err);
	auto root2 = distr::get_iosvc(*manager).lookup_node(err, "root2");
	ASSERT_NOERR(err);

	EXPECT_ARRHAS(graph_roots, root1ref);
	EXPECT_ARRHAS(graph_roots, root2);

	std::stringstream ss;
	distr::get_printsvc(*manager2).print_ascii(ss, root1.get());
	std::string expect =
		"(SUB)\n"
		"_`--(variable:src2)\n"
		"_`--[mgr]:(POW)\n"
		"_____`--[mgr2]:(DIV)\n"
		"_____|___`--[mgr]:(NEG)\n"
		"_____|___|___`--(variable:osrc)\n"
		"_____|___`--(ADD)\n"
		"_____|_______`--(SIN)\n"
		"_____|_______|___`--(variable:src)\n"
		"_____|_______`--(variable:src)\n"
		"_____`--(variable:osrc2)\n";
	EXPECT_STREQ(expect.c_str(), ss.str().c_str());

	std::stringstream ss2;
	distr::get_printsvc(*manager).print_ascii(ss2, root2.get());
	std::string expect2 =
		"(SUB)\n"
		"_`--(variable:s2src)\n"
		"_`--[mgr2]:(MUL)\n"
		"_____`--[mgr]:(ABS)\n"
		"_____|___`--(variable:s2src)\n"
		"_____`--(EXP)\n"
		"_____|___`--(variable:s2src2)\n"
		"_____`--(NEG)\n"
		"_________`--(variable:s2src3)\n";
	EXPECT_STREQ(expect2.c_str(), ss2.str().c_str());
}


#endif // DISABLE_OXSVC_LOAD_TEST
