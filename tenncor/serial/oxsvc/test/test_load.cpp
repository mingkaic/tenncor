
#ifndef DISABLE_OXSVC_LOAD_TEST


#include <fstream>

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/global/mock/mock.hpp"

#include "dbg/print/teq.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/serial/oxsvc/oxsvc.hpp"


const std::string testdir = "models/test";


const std::string test_service = "tenncor.serial.oxsvc.test";


struct LOAD : public ::testing::Test, public DistrTestcase
{
	LOAD (void) : DistrTestcase(test_service) {}

protected:
	void TearDown (void) override
	{
		clean_up();
	}

	distr::iDistrMgrptrT make_mgr (size_t port, const std::string& id = "")
	{
		return DistrTestcase::make_mgr(port, {
			distr::register_iosvc,
			distr::register_oxsvc,
		}, id);
	}

	void check_clean (void)
	{
		ppconsul::catalog::Catalog catalog(*consul_);
		auto services = catalog.service(service_name_);
		ASSERT_EQ(services.size(), 0);
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
	auto& svc2 = distr::get_iosvc(*manager2);

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

	EXPECT_GRAPHEQ(
		"(SUB<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(variable:src2<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(POW<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(placeholder:mgr2/2<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:osrc2<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n", root1);
	EXPECT_GRAPHEQ(
		"(SUB<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(placeholder:mgr2/1<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(placeholder:mgr2/3<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXP<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:s2src2<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(NEG<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(placeholder:mgr2/4<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n", root2);

	auto refs = distr::reachable_refs(teq::TensptrsT{root1});
	auto refs2 = distr::reachable_refs(teq::TensptrsT{root2});

	ASSERT_EQ(1, refs.size());
	ASSERT_EQ(3, refs2.size());

	types::StrUSetT refids = {"1", "3", "4"};
	distr::iDistrRef* p1_ref;
	distr::iDistrRef* p3_ref;
	distr::iDistrRef* p4_ref;
	for (auto ref : refs2)
	{
		auto id = ref->node_id();
		ASSERT_HAS(refids, id);
		if (id == "1")
		{
			p1_ref = ref;
		}
		else if (id == "3")
		{
			p3_ref = ref;
		}
		else if (id == "4")
		{
			p4_ref = ref;
		}
	}

	error::ErrptrT err = nullptr;
	auto p2 = svc2.lookup_node(err, (*refs.begin())->node_id());
	ASSERT_NOERR(err);
	auto p1 = svc2.lookup_node(err, p1_ref->node_id());
	ASSERT_NOERR(err);
	auto p3 = svc2.lookup_node(err, p3_ref->node_id());
	ASSERT_NOERR(err);
	auto p4 = svc2.lookup_node(err, p4_ref->node_id());
	ASSERT_NOERR(err);

	EXPECT_GRAPHEQ(
		"(ABS<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(variable:s2src<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n", p2);

	EXPECT_GRAPHEQ(
		"(DIV<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(NEG<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(variable:osrc<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_`--(ADD<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(SIN<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:src<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:src<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n", p1);

	EXPECT_GRAPHEQ(
		"(variable:s2src<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n", p3);

	EXPECT_GRAPHEQ(
		"(variable:s2src3<DOUBLE>[3\\7\\1\\1\\1\\1\\1\\1])\n", p4);
}


#endif // DISABLE_OXSVC_LOAD_TEST
