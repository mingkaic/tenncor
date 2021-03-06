
#ifndef DISABLE_IOSVC_LOOKUP_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"
#include "tenncor/distr/mock/mock.hpp"
#include "tenncor/distr/iosvc/mock/mock.hpp"

#include "tenncor/distr/iosvc/iosvc.hpp"


const std::string test_service = "tenncor.distr.iosvc.test";


struct LOOKUP : public ::testing::Test, public DistrTestcase
{
protected:
	void TearDown (void) override
	{
		MockServerBuilder::clear_service();
	}

	distr::iDistrMgrptrT make_mgr (const std::string& id)
	{
		return make_mgr(id, reserve_port());
	}

	distr::iDistrMgrptrT make_mgr (const std::string& id, size_t port)
	{
		return DistrTestcase::make_local_mgr(port, {
			register_mock_iosvc,
		}, id);
	}
};


TEST_F(LOOKUP, LookupId)
{
	distr::iDistrMgrptrT mgr(make_mgr("mgr"));
	auto& service = distr::get_iosvc(*mgr);

	teq::Shape outshape({2, 2});
	std::vector<double> data{2, 3, 7, 2};
	MockDeviceRef devref;
	auto a = make_var(data.data(), devref, outshape);
	auto b = make_var(data.data(), devref, outshape);
	service.expose_node(a);

	auto ida = service.lookup_id(a.get());
	auto idb = service.lookup_id(b.get());
	EXPECT_TRUE(ida);
	EXPECT_FALSE(idb);
}


TEST_F(LOOKUP, LocalLookupNode)
{
	distr::iDistrMgrptrT mgr(make_mgr("mgr"));
	auto& service = distr::get_iosvc(*mgr);

	teq::Shape outshape({2, 2});
	std::vector<double> data{2, 3, 7, 2};
	MockDeviceRef devref;
	auto a = make_var(data.data(), devref, outshape);
	service.expose_node(a);
	auto ida = *service.lookup_id(a.get());

	error::ErrptrT err = nullptr;
	auto refa = service.lookup_node(err, ida);
	ASSERT_EQ(a, refa);
	ASSERT_NOERR(err);
}


TEST_F(LOOKUP, RemoteLookupNode)
{
	distr::iDistrMgrptrT manager(make_mgr("mgr"));
	auto& service = distr::get_iosvc(*manager);

	teq::Shape outshape({2, 2});
	std::vector<double> data{2, 3, 7, 2};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto a = make_var(data.data(), devref, outshape);
	EXPECT_CALL(*a, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	service.expose_node(a);
	auto ida = *service.lookup_id(a.get());

	distr::iDistrMgrptrT manager2(make_mgr("mgr2"));
	auto& service2 = distr::get_iosvc(*manager2);

	error::ErrptrT err = nullptr;
	EXPECT_EQ(nullptr, service2.lookup_node(err, ida, false));
	ASSERT_NE(nullptr, err);
	auto expect_msg = fmts::sprintf(
		"no id %s found locally: will not recurse", ida.c_str());
	EXPECT_STREQ(expect_msg.c_str(), err->to_string().c_str());

	err = nullptr;
	auto refa = service2.lookup_node(err, ida);
	ASSERT_NOERR(err);
	ASSERT_NE(nullptr, refa);
	auto expect_refname = fmts::sprintf("mgr/%s", ida.c_str());
	EXPECT_STREQ(expect_refname.c_str(), refa->to_string().c_str());
}


TEST_F(LOOKUP, ExposingReferences)
{
	distr::iDistrMgrptrT mgr(make_mgr("mgr"));
	auto& service = distr::get_iosvc(*mgr);

	distr::iDistrMgrptrT manager2(make_mgr("mgr2"));
	auto& service2 = distr::get_iosvc(*manager2);

	teq::Shape outshape({2, 2});
	std::vector<double> data{2, 3, 7, 2};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto a = make_var(data.data(), devref, outshape);
	EXPECT_CALL(*a, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));

	auto aid = service.expose_node(a);

	error::ErrptrT err = nullptr;
	auto ref = service2.lookup_node(err, aid);
	ASSERT_NOERR(err);

	auto refid = service.expose_node(ref);

	EXPECT_STREQ(aid.c_str(), refid.c_str());
}


#endif // DISABLE_IOSVC_LOOKUP_TEST
