
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

	teq::Shape shape;
	auto c = make_var(shape, "3.5");
	auto x = make_var(shape, "X");
	auto y = make_var(shape, "Y");
	auto f = make_fnc("ADD", 0, teq::TensptrsT{c,y});
	auto f1 = make_fnc("SIN", 0, teq::TensptrsT{c});
	auto f2 = make_fnc("ADD", 0, teq::TensptrsT{x,x});
	auto f3 = make_fnc("MUL", 0, teq::TensptrsT{f1,f2});
	auto root = make_fnc("ADD", 0, teq::TensptrsT{f,f3});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f1, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f2, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f3, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*root, shape()).WillRepeatedly(Return(shape));

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

	teq::Shape shape;
	auto c = make_var(shape, "3.5");
	auto x = make_var(shape, "X");
	auto y = make_var(shape, "Y");

	MockMeta mockmeta;
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));

	EXPECT_CALL(*c, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*x, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*y, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	auto f = make_fnc("SIN", 0, teq::TensptrsT{c});
	auto g = make_fnc("ADD", 0, teq::TensptrsT{x,x});
	auto f1 = make_fnc("ADD", 0, teq::TensptrsT{c,y});
	auto f2 = make_fnc("MUL", 0, teq::TensptrsT{f,g});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*g, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f1, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f2, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*g, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*f1, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*f2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	auto& iosvc = distr::get_iosvc(*mgr);
	auto& iosvc2 = distr::get_iosvc(*mgr2);
	std::string f1_id = iosvc2.expose_node(f1);
	std::string f2_id = iosvc2.expose_node(f2);

	error::ErrptrT err = nullptr;
	auto f1_ref = iosvc.lookup_node(err, f1_id);
	ASSERT_NOERR(err);
	auto f2_ref = iosvc.lookup_node(err, f2_id);
	ASSERT_NOERR(err);

	auto root = make_fnc("ADD", 0, teq::TensptrsT{f1_ref, f2_ref});
	EXPECT_CALL(*root, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*root, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

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
