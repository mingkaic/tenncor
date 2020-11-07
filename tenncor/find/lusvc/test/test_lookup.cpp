
#ifndef DISABLE_LUSVC_LOOKUP_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/mock/mock.hpp"
#include "tenncor/distr/iosvc/mock/mock.hpp"
#include "tenncor/find/lusvc/mock/mock.hpp"


const std::string test_service = "tenncor.find.lusvc.test";


struct LOOKUP : public ::testing::Test, public DistrTestcase
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
			register_mock_lusvc,
		}, id);
	}
};


TEST_F(LOOKUP, LocalLookup)
{
	// instance 1
	distr::iDistrMgrptrT mgr(make_mgr("mgr1"));

	auto c = std::make_shared<MockLeaf>(teq::Shape(), "3.5");
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto y = std::make_shared<MockLeaf>(teq::Shape(), "Y");
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{
		std::make_shared<MockFunctor>(teq::TensptrsT{c,y}, teq::Opcode{"ADD", 0}),
		std::make_shared<MockFunctor>(teq::TensptrsT{
			std::make_shared<MockFunctor>(teq::TensptrsT{c}, teq::Opcode{"SIN", 0}),
			std::make_shared<MockFunctor>(teq::TensptrsT{x,x}, teq::Opcode{"ADD", 0})
		}, teq::Opcode{"MUL", 0})
	}, teq::Opcode{"ADD", 0});

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"ADD\","
			"\"args\":[{\"cst\":3.5},{\"symb\":\"B\"}]"
		"}}";
	query::Node cond;
	query::json_parse(cond, condjson);

	auto& svc = distr::get_lusvc(*mgr);
	auto detections = svc.query({root}, cond);
	ASSERT_EQ(1, detections.size());
	teq::TensptrT detection = detections.front();
	char expected[] =
		"(ADD)\n"
		"_`--(constant:3.5)\n"
		"_`--(constant:Y)\n";

	PrettyEquation peq;
	std::stringstream ss;
	peq.print(ss, detection);
	EXPECT_STREQ(expected, ss.str().c_str());
}


TEST_F(LOOKUP, RemoteLookup)
{
	// instance 1
	distr::iDistrMgrptrT mgr(make_mgr("mgr1"));

	// instance 2
	distr::iDistrMgrptrT mgr2(make_mgr("mgr2"));

	auto c = std::make_shared<MockLeaf>(teq::Shape(), "3.5");
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto y = std::make_shared<MockLeaf>(teq::Shape(), "Y");

	c->meta_.tcode_ = egen::DOUBLE;
	x->meta_.tcode_ = egen::DOUBLE;
	y->meta_.tcode_ = egen::DOUBLE;
	c->meta_.tname_ = "DOUBLE";
	x->meta_.tname_ = "DOUBLE";
	y->meta_.tname_ = "DOUBLE";

	auto f = std::make_shared<MockFunctor>(teq::TensptrsT{c}, teq::Opcode{"SIN", 0});
	auto g = std::make_shared<MockFunctor>(teq::TensptrsT{x,x}, teq::Opcode{"ADD", 0});
	auto f1 = std::make_shared<MockFunctor>(teq::TensptrsT{c,y}, teq::Opcode{"ADD", 0});
	auto f2 = std::make_shared<MockFunctor>(teq::TensptrsT{f,g}, teq::Opcode{"MUL", 0});
	f->meta_.tcode_ = egen::DOUBLE;
	g->meta_.tcode_ = egen::DOUBLE;
	f1->meta_.tcode_ = egen::DOUBLE;
	f2->meta_.tcode_ = egen::DOUBLE;
	f->meta_.tname_ = "DOUBLE";
	g->meta_.tname_ = "DOUBLE";
	f1->meta_.tname_ = "DOUBLE";
	f2->meta_.tname_ = "DOUBLE";

	auto& iosvc = distr::get_iosvc(*mgr);
	auto& iosvc2 = distr::get_iosvc(*mgr2);
	std::string f1_id = iosvc2.expose_node(f1);
	std::string f2_id = iosvc2.expose_node(f2);

	error::ErrptrT err = nullptr;
	auto f1_ref = iosvc.lookup_node(err, f1_id);
	ASSERT_NOERR(err);
	auto f2_ref = iosvc.lookup_node(err, f2_id);
	ASSERT_NOERR(err);

	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{f1_ref, f2_ref}, teq::Opcode{"ADD", 0});
	root->meta_.tcode_ = egen::DOUBLE;
	root->meta_.tname_ = "DOUBLE";

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"ADD\","
			"\"args\":[{\"cst\":3.5},{\"symb\":\"B\"}]"
		"}}";
	query::Node cond;
	query::json_parse(cond, condjson);

	auto& svc = distr::get_lusvc(*mgr);
	auto detections = svc.query({root}, cond);
	ASSERT_EQ(1, detections.size());
	auto result = detections.front();

	auto ref = dynamic_cast<distr::iDistrRef*>(result.root_.get());
	ASSERT_NE(nullptr, ref);
	EXPECT_STREQ("mgr2", ref->cluster_id().c_str());
	EXPECT_STREQ(f1_id.c_str(), ref->node_id().c_str());

	auto bid = iosvc2.lookup_id(y.get());
	ASSERT_TRUE(bid);
	ASSERT_EQ(1, result.symbs_.size());
	ASSERT_HAS(result.symbs_, "B");
	auto bref = dynamic_cast<distr::iDistrRef*>(result.symbs_.at("B").get());
	ASSERT_NE(nullptr, bref);
	EXPECT_STREQ("mgr2", bref->cluster_id().c_str());
	EXPECT_STREQ(bid->c_str(), bref->node_id().c_str());
}


#endif // DISABLE_LUSVC_LOOKUP_TEST
