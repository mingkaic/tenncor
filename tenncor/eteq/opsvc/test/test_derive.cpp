
#ifndef DISABLE_OPSVC_DERIVE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/mock/mock.hpp"
#include "tenncor/distr/iosvc/mock/mock.hpp"
#include "tenncor/eteq/opsvc/mock/mock.hpp"


using ::testing::Invoke;
using ::testing::DoAll;
using ::testing::SaveArg;


const std::string test_service = "tenncor.eteq.opsvc.test";


struct DERIVE : public ::testing::Test, public DistrTestcase
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
			[](estd::ConfigMap<>& svcs, const distr::PeerServiceConfig& cfg) -> error::ErrptrT
			{
				auto iosvc = static_cast<distr::io::DistrIOService*>(svcs.get_obj(distr::io::iosvc_key));
				if (nullptr == iosvc)
				{
					return error::error("opsvc requires iosvc already registered");
				}
				svcs.add_entry<distr::op::DistrOpService>(distr::op::opsvc_key,
				[&]
				{
					return new distr::op::DistrOpService(
						std::make_unique<eigen::Device>(std::numeric_limits<size_t>::max()),
						std::make_unique<MockDerivativeFunc>(), cfg, iosvc,
						std::make_shared<MockDistrOpCliBuilder>(),
						std::make_shared<MockOpService>());
				});
				return nullptr;
			},
		}, id);
	}
};


TEST_F(DERIVE, RemoteDerivation)
{
	teq::Shape shape({2, 2});
	std::vector<double> data = {1, 2, 3, 4};
	MockDeviceRef devref;
	MockMeta mockmeta;

	// instance 1
	distr::iDistrMgrptrT mgr(make_mgr("mgr1"));

	auto a = make_var(data.data(), devref, shape, "a");
	auto b = make_var(data.data(), devref, shape, "b");
	auto c = make_var(data.data(), devref, shape, "c");
	EXPECT_CALL(*a, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*b, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*c, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));

	auto d = make_fnc("FUNC5", 0, teq::TensptrsT{b, a});
	auto e = make_fnc("FUNC4", 0, teq::TensptrsT{c, d});
	auto farg = make_fnc("FUNC3", 0, teq::TensptrsT{d});
	auto farg2 = make_fnc("FUNC2", 0, teq::TensptrsT{c});
	auto f = make_fnc("FUNC1", 0, teq::TensptrsT{farg, farg2});
	EXPECT_CALL(*d, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*e, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*farg, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*farg2, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*d, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*e, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*farg, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*farg2, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*f, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(*d), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(*e), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(*farg), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(*farg2), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(*f), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*d, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*e, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*farg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*farg2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*f, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	auto root = make_fnc("FUNC", 0, teq::TensptrsT{e, f});
	EXPECT_CALL(*root, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*root, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(*root), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(*root, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	auto& svc = distr::get_iosvc(*mgr);
	std::string root_id = svc.expose_node(root);
	std::string base_id = svc.expose_node(a);

	// instance 2
	distr::iDistrMgrptrT mgr2 = make_mgr("mgr2");
	auto& svc2 = distr::get_iosvc(*mgr2);

	error::ErrptrT err = nullptr;
	teq::TensptrT root_ref = svc2.lookup_node(err, root_id);
	ASSERT_NOERR(err);
	teq::TensptrT base_ref = svc2.lookup_node(err, base_id);
	ASSERT_NOERR(err);

	MockDerivativeFunc* remotedfunc = dynamic_cast<MockDerivativeFunc*>(
		distr::get_opsvc(*mgr).get_derivfuncs());
	MockDerivativeFunc* localdfunc = dynamic_cast<MockDerivativeFunc*>(
		distr::get_opsvc(*mgr2).get_derivfuncs());
	ASSERT_NE(nullptr, remotedfunc);
	ASSERT_NE(nullptr, localdfunc);

	auto der = make_var(shape, "der");
	auto der2 = make_var(shape, "der2");
	auto der3 = make_var(shape, "der3");
	auto der4 = make_var(shape, "der4");
	auto der5 = make_var(shape, "der5");
	auto der6 = make_var(shape, "der6");
	auto der7 = make_var(shape, "der7");
	auto der8 = make_var(shape, "der8");
	EXPECT_CALL(*der, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*der2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*der3, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*der4, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*der5, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*der6, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*der7, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*der8, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	teq::TensptrT capder = nullptr;
	teq::TensptrT cap2der = nullptr;
	EXPECT_CALL(*remotedfunc, lderive(teq::FuncptrT(root), _, 0)).Times(1).
		WillOnce(DoAll(SaveArg<1>(&capder),Return(der2)));
	EXPECT_CALL(*remotedfunc, lderive(teq::FuncptrT(root), _, 1)).Times(1).
		WillOnce(DoAll(SaveArg<1>(&cap2der),Return(der3)));
	EXPECT_CALL(*remotedfunc, lderive(teq::FuncptrT(f), teq::TensptrT(der3), 0)).Times(1).
		WillOnce(Return(der4));
	EXPECT_CALL(*remotedfunc, lderive(teq::FuncptrT(e), teq::TensptrT(der2), 1)).Times(1).
		WillOnce(Return(der5));
	EXPECT_CALL(*remotedfunc, lderive(teq::FuncptrT(farg), teq::TensptrT(der4), 0)).Times(1).
		WillOnce(Return(der6));
	EXPECT_CALL(*remotedfunc, lderive(teq::FuncptrT(d), teq::TensptrT(der8), 1)).Times(1).
		WillOnce(Return(der7));
	EXPECT_CALL(*remotedfunc, add(teq::TensptrsT{der5, der6})).Times(1).WillOnce(Return(der8));

	teq::TensMapT<teq::TensptrsT> grads = {
		{root_ref.get(), {der}}
	};
	auto remote_ders = distr::get_opsvc(*mgr2).derive(
		grads, {root_ref}, distr::op::BackpropMeta{
			teq::TensSetT{base_ref.get()}});

	ASSERT_EQ(1, remote_ders.size());
	EXPECT_HAS(remote_ders, base_ref.get());
	auto dbase = remote_ders[base_ref.get()];

	// compare der reference after to ensure that derive automatically references der
	EXPECT_EQ(capder, cap2der);
	auto derid = svc2.lookup_id(der.get());
	ASSERT_TRUE(bool(derid));
	auto derref = svc.lookup_node(err, *derid);
	ASSERT_NOERR(err);
	EXPECT_EQ(derref, capder);

	auto der7id = svc.lookup_id(der7.get());
	ASSERT_TRUE(bool(der7id));
	auto der7ref = svc2.lookup_node(err, *der7id);
	ASSERT_NOERR(err);
	EXPECT_EQ(der7ref, dbase);
}


#endif // DISABLE_OPSVC_DERIVE_TEST
