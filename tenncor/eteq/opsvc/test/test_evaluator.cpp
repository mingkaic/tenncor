
#ifndef DISABLE_OPSVC_EVALUATE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/mock/mock.hpp"
#include "tenncor/distr/iosvc/mock/mock.hpp"
#include "tenncor/eteq/opsvc/mock/mock.hpp"


using ::testing::Invoke;


const std::string test_service = "tenncor.eteq.opsvc.test";


struct EVALUATE : public ::testing::Test, public DistrTestcase
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
						std::make_unique<MockDevice>(),
						std::make_unique<eteq::DerivativeFuncs>(), cfg, iosvc,
						std::make_shared<MockDistrOpCliBuilder>(),
						std::make_shared<MockOpService>());
				});
				return nullptr;
			},
		}, id);
	}
};


TEST_F(EVALUATE, SimpleGraph)
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
	std::vector<double> exdata = {
		2997, 3430, 68, 10626, 4956, 8241,
		5016, 4956, 4554, 13156, 12384, 2088,
		4895, 4620, 2480, 5832, 2240, 2430,
		2100, 7140, 8908, 520, 2940, 4600,
	};

	// instance 1
	distr::iDistrMgrptrT mgr(make_mgr("mgr1"));

	MockMeta mockmeta;
	MockDeviceRef devref;
	MockDeviceRef devref2;
	auto src = make_var(data.data(), devref, shape, "src");
	auto src2 = make_var(data2.data(), devref2, shape, "src2");
	auto dest = make_fnc("ADD", 7, teq::TensptrsT{src, src2});
	EXPECT_CALL(*dest, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*dest, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	auto& svc = distr::get_iosvc(*mgr);
	std::string id = svc.expose_node(dest);

	// instance 2
	distr::iDistrMgrptrT mgr2(make_mgr("mgr2"));

	MockDeviceRef devref3;
	auto src3 = make_var(data3.data(), devref3, shape, "src3");
	auto& svc2 = distr::get_iosvc(*mgr2);
	error::ErrptrT err = nullptr;
	auto ref = svc2.lookup_node(err, id);
	ASSERT_NOERR(err);
	ASSERT_NE(nullptr, ref);
	auto dest2 = make_fnc("MUL", 8, teq::TensptrsT{ref, src3});
	EXPECT_CALL(*dest2, shape()).WillRepeatedly(Return(shape));

	// before
	// * (dest2) = not updated
	// `-- mgr1:+ (ref/dest) = not updated
	// |   `-- src
	// |   `-- src2
	// `-- src3

	// after
	// * (dest2) = updated
	// `-- mgr1:+ (ref/dest) = updated
	// |   `-- src
	// |   `-- src2
	// `-- src3

	MockDeviceRef destdev;
	EXPECT_CALL(mockmeta, state_version()).WillOnce(Return(0));
	EXPECT_CALL(*dest, device()).WillOnce(ReturnRef(destdev));
	EXPECT_CALL(destdev, data()).Times(1).WillOnce(Return(data.data()));

	const teq::iTensor* capdest = nullptr;
	const teq::iTensor* capdest2 = nullptr;
	MockDevice* localdev = dynamic_cast<MockDevice*>(distr::get_opsvc(*mgr2).get_evaluator());
	MockDevice* foreigndev = dynamic_cast<MockDevice*>(distr::get_opsvc(*mgr).get_evaluator());
	ASSERT_NE(nullptr, localdev);
	ASSERT_NE(nullptr, foreigndev);
	EXPECT_CALL(*localdev, calc(_)).Times(1).
		WillOnce(Invoke([&capdest2](const teq::iTensor& args){ capdest2 = &args; }));
	EXPECT_CALL(*foreigndev, calc(_)).Times(1).
		WillOnce(Invoke([&capdest](const teq::iTensor& args){ capdest = &args; }));
	distr::get_opsvc(*mgr2).evaluate(*localdev, {dest2.get()});

	EXPECT_EQ(dest.get(), capdest);
	EXPECT_EQ(dest2.get(), capdest2);
}


#endif // DISABLE_OPSVC_EVALUATE_TEST
