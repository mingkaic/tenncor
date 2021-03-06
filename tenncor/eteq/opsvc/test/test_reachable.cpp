
#ifndef DISABLE_OPSVC_REACHABLE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/mock/mock.hpp"
#include "tenncor/distr/iosvc/mock/mock.hpp"
#include "tenncor/eteq/opsvc/mock/mock.hpp"


const std::string test_service = "tenncor.eteq.opsvc.test";


struct REACHABLE : public ::testing::Test, public DistrTestcase
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
			register_mock_opsvc,
		}, id);
	}
};


TEST_F(REACHABLE, CyclicGraph)
{
	teq::Shape shape({2, 3, 4});
	std::vector<double> data = {
		63, 19, 11, 94, 23, 63,
		3, 48, 60, 77, 62, 32,
		35, 89, 33, 64, 36, 64,
		25, 49, 41, 1, 4, 97,
	};
	std::vector<double> data2 = {
		18, 30, 23, 60, 36, 60,
		73, 36, 6, 66, 67, 84,
		54, 43, 29, 8, 20, 71,
		10, 53, 90, 7, 94, 87,
	};
	std::vector<double> data3 = {
		37, 70, 2, 69, 84, 67,
		66, 59, 69, 92, 96, 18,
		55, 35, 40, 81, 40, 18,
		60, 70, 68, 65, 30, 25,
	};

	distr::iDistrMgrptrT mgrA(make_mgr("mgrA"));
	distr::iDistrMgrptrT mgrB(make_mgr("mgrB"));
	distr::iDistrMgrptrT mgrC(make_mgr("mgrC"));
	distr::iDistrMgrptrT mgrD(make_mgr("mgrD"));
	distr::iDistrMgrptrT mgrE(make_mgr("mgrE"));
	distr::iDistrMgrptrT mgrF(make_mgr("mgrF"));

	MockMeta mockmeta;
	MockDeviceRef devref;
	MockDeviceRef devref2;
	MockDeviceRef devref3;
	auto f1 = make_var(data.data(), devref, shape, "f1");
	auto f2 = make_var(data2.data(), devref2, shape, "f2");
	auto d1 = make_var(data3.data(), devref3, shape, "e3");

	auto e2 = make_var(data.data(), devref, shape, "e2");
	auto e3 = make_var(data2.data(), devref2, shape, "e3");
	auto e1 = make_fnc("ADD", 7, teq::TensptrsT{e2, e3});
	EXPECT_CALL(*e1, shape()).WillRepeatedly(Return(shape));

	EXPECT_CALL(*e1, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*f1, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*f2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*d1, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));

	distr::get_iosvc(*mgrE).expose_node(e1);
	distr::get_iosvc(*mgrF).expose_node(f1);
	distr::get_iosvc(*mgrF).expose_node(f2);
	distr::get_iosvc(*mgrD).expose_node(d1);

	std::string e1_key = *distr::get_iosvc(*mgrE).lookup_id(e1.get());

	error::ErrptrT err = nullptr;
	std::string f1_key = *distr::get_iosvc(*mgrF).lookup_id(f1.get());
	auto f1_ref = distr::get_iosvc(*mgrA).lookup_node(err, f1_key);
	ASSERT_NOERR(err);
	auto a2 = make_fnc("NEG", 6, teq::TensptrsT{f1_ref});
	EXPECT_CALL(*a2, shape()).WillRepeatedly(Return(shape));

	EXPECT_CALL(*a2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	distr::get_iosvc(*mgrA).expose_node(a2);

	std::string d1_key = *distr::get_iosvc(*mgrD).lookup_id(d1.get());
	auto a2_ref = distr::get_iosvc(*mgrC).lookup_node(err, *distr::get_iosvc(*mgrA).lookup_id(a2.get()));
	ASSERT_NOERR(err);
	auto d1_ref = distr::get_iosvc(*mgrC).lookup_node(err, d1_key);
	ASSERT_NOERR(err);
	auto c1 = make_fnc("ADD", 7, teq::TensptrsT{a2_ref, d1_ref});
	EXPECT_CALL(*c1, shape()).WillRepeatedly(Return(shape));

	EXPECT_CALL(*c1, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	distr::get_iosvc(*mgrC).expose_node(c1);

	auto c1_ref = distr::get_iosvc(*mgrB).lookup_node(err, *distr::get_iosvc(*mgrC).lookup_id(c1.get()));
	ASSERT_NOERR(err);
	auto f2_ref = distr::get_iosvc(*mgrB).lookup_node(err, *distr::get_iosvc(*mgrF).lookup_id(f2.get()));
	ASSERT_NOERR(err);
	auto b1 = make_fnc("MUL", 8, teq::TensptrsT{c1_ref, f2_ref});
	EXPECT_CALL(*b1, shape()).WillRepeatedly(Return(shape));

	EXPECT_CALL(*b1, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	distr::get_iosvc(*mgrB).expose_node(b1);

	auto b1_ref = distr::get_iosvc(*mgrA).lookup_node(err, *distr::get_iosvc(*mgrB).lookup_id(b1.get()));
	ASSERT_NOERR(err);
	auto a1 = make_fnc("SIN", 5, teq::TensptrsT{b1_ref});
	EXPECT_CALL(*a1, shape()).WillRepeatedly(Return(shape));

	EXPECT_CALL(*a1, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	distr::get_iosvc(*mgrA).expose_node(a1);

	// able to reach across cyclical graph
	auto reachables = estd::map_keyset(distr::get_opsvc(*mgrA).reachable(err, {a1.get()}, {f1_key}));
	EXPECT_EQ(1, reachables.size());
	EXPECT_EQ(a1.get(), *reachables.begin());

	// unable to reach isolated nodes
	auto nonreachable = distr::get_opsvc(*mgrA).reachable(err, {a1.get()}, {e1_key});
	EXPECT_TRUE(nonreachable.empty());

	// unable to reach node unreachable nodes
	auto nonreachable2 = distr::get_opsvc(*mgrA).reachable(err, {a2.get()}, {d1_key});
	EXPECT_TRUE(nonreachable2.empty());
}


#endif // DISABLE_OPSVC_REACHABLE_TEST
